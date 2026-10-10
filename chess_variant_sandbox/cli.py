"""Command line entry point for every stage of the pipeline.

One CLI rather than a script per stage, because the stages share vocabulary -- a
ruleset path, an agent list, a run directory -- and the design loop drives the same
operations programmatically. Every subcommand that produces output writes it under
``runs/`` keyed by ruleset hash, so a result can always be traced back to the exact
rules that produced it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chess_variant_sandbox import corpus
from chess_variant_sandbox.agents import available_agents
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.measurement import stale_build
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS, score_ruleset
from chess_variant_sandbox.spec import load_spec, measure_complexity, spec_hash, validate_spec
from chess_variant_sandbox.tournament import (
    RatingTable,
    TournamentConfig,
    play_game,
    rate,
    read_records,
)
from chess_variant_sandbox.tournament.analysis import BLUNDER_SAMPLE

if TYPE_CHECKING:
    from chess_variant_sandbox.tournament.fitting import Reproduction

DEFAULT_AGENTS = ("random", "greedy", "minimax:nodes=800", "minimax:nodes=3200")
RUNS_DIR = Path("runs")
# Where `chess-variant-sandbox tournament` writes when it is given no `--out`: beside the library, never
# in it. A library game's runs are `runs/<hash>-<seed>` and its measurement `runs/<hash>`,
# and the old default was the first of those -- so a one-off at seed 0 on another roster
# replaced a library run's games without staling the file that scored them (Q335). One
# directory down, the name is not a hash and the library's layout has no slot inside it.
TOURNAMENT_DIR = RUNS_DIR / "tournament"
# How many games the verdict writer is shown. Named here rather than imported, because
# `chess_variant_sandbox.llm` pulls in the SDK and the CLI must parse its arguments without it.
JUDGE_SAMPLE = 4


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 1
    try:
        if args.checkpoint is None:
            return args.handler(args)
        # Q732: a campaign directory; every long stage of the command is resumable there.
        from chess_variant_sandbox.tournament import checkpoint

        command = list(sys.argv[1:] if argv is None else argv)
        with checkpoint.active(args.checkpoint, args.checkpoint_minutes, command=command):
            code: int = args.handler(args)
        return code
    except (ValueError, LookupError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chess-variant-sandbox", description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="DIR",
        help="a campaign directory: every schedule, harvest and training the command runs is "
        "written there in pieces as they finish, and the same command run again resumes "
        "from what is on disk, to the same result (tournament/checkpoint.py)",
    )
    parser.add_argument(
        "--checkpoint-minutes",
        type=float,
        default=10.0,
        metavar="M",
        help="with --checkpoint: how often a training run saves its state (default 10)",
    )
    sub = parser.add_subparsers()

    validate = sub.add_parser("validate", help="check a ruleset against the schema and compile it")
    validate.add_argument("ruleset", type=Path)
    validate.set_defaults(handler=_cmd_validate)

    bench = sub.add_parser("bench", help="measure move generation throughput")
    bench.add_argument("ruleset", type=Path)
    bench.add_argument("--depth", type=int, default=4)
    bench.add_argument("--games", type=int, default=200)
    bench.add_argument("--threads", type=int, default=1, help="1 keeps the number comparable")
    bench.set_defaults(handler=_cmd_bench)

    play = sub.add_parser("play", help="play a single game and print it")
    play.add_argument("ruleset", type=Path)
    # Two values rather than one colon-separated string: agent configs contain colons
    # themselves ("minimax:nodes=2000"), so any single-string form splits in the wrong
    # place and silently plays the wrong match.
    play.add_argument("--agents", nargs=2, default=["greedy", "random"], metavar="AGENT")
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--quiet", action="store_true")
    play.set_defaults(handler=_cmd_play)

    tournament = sub.add_parser("tournament", help="run a full round robin")
    tournament.add_argument("ruleset", type=Path)
    tournament.add_argument("--agents", nargs="*", default=list(DEFAULT_AGENTS))
    tournament.add_argument("--games", type=int, default=4, help="games per ordered pairing")
    tournament.add_argument("--seed", type=int, default=0)
    tournament.add_argument("--workers", type=int, default=0, help="threads; 0 means all but one core")
    tournament.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"directory for games.jsonl and tournament.json (default "
             f"{TOURNAMENT_DIR.as_posix()}/<hash>-<seed>, which the same ruleset and seed "
             "replace on the next run; never a library run, which `chess-variant-sandbox measure` writes)",
    )
    tournament.set_defaults(handler=_cmd_tournament)

    score = sub.add_parser("score", help="compute the interestingness index for a finished run")
    score.add_argument("run", type=Path)
    score.add_argument("--ruleset", type=Path, default=None)
    score.add_argument("--weights", default=DEFAULT_WEIGHTS)
    score.add_argument("--json", action="store_true")
    score.add_argument(
        "--no-interval",
        action="store_true",
        help="skip the bootstrap error bar (faster, and much easier to misread)",
    )
    score.set_defaults(handler=_cmd_score)

    analyse = sub.add_parser(
        "analyse", help="run the engine-derived second pass over a finished run"
    )
    analyse.add_argument("run", help="run directory containing games.jsonl")
    analyse.add_argument("--ruleset", help="spec to replay against, if not discoverable")
    # The constant itself, never a copy of its value. `analyse` rewrites `analysis.json`
    # in place and the measurement basis cannot see which sample it was drawn on, so a
    # hand-run pass at a different count silently replaces a campaign's numbers with
    # numbers taken another way -- which is what made `v5`'s `blunder_recoverability`
    # look like it had moved when nothing had (`docs/archive/bugs-resolved.md`, the `v5` entry).
    analyse.add_argument(
        "--blunders",
        type=int,
        default=BLUNDER_SAMPLE,
        help=f"blunder trials to run (default {BLUNDER_SAMPLE}, what every campaign uses)",
    )
    analyse.add_argument("--workers", type=int, default=0)
    analyse.set_defaults(handler=_cmd_analyse)

    heur = sub.add_parser(
        "heuristics",
        help="generate or play the heuristic players for a ruleset",
    )
    heur.add_argument("ruleset", help="ruleset file to build heuristics for")
    heur.add_argument(
        "--source",
        default="handwritten",
        choices=("handwritten", "random", "model"),
        help="where the heuristics come from; 'model' needs credentials",
    )
    heur.add_argument("--k", type=int, default=6, help="how many heuristics, anchor included")
    heur.add_argument("--store", default="heuristics", help="directory of heuristic sets")
    heur.add_argument("--games", type=int, default=3, help="games per ordered pairing")
    heur.add_argument("--nodes", type=int, default=0, help="deep budget; 0 chooses it")
    heur.add_argument("--seed", type=int, default=0)
    heur.add_argument("--workers", type=int, default=0)
    heur.add_argument("--out", default="", help="where to write the report")
    heur.add_argument(
        "--generate-only", action="store_true", help="write the set and play nothing"
    )
    heur.add_argument(
        "--against-fit",
        action="store_true",
        help="measure agreement against this ruleset's fitted objective, not the built-in one",
    )
    heur.set_defaults(handler=_cmd_heuristics)

    fit = sub.add_parser(
        "fit",
        help="fit this ruleset's objective from games, and check it against the built-in one",
    )
    fit.add_argument("ruleset", help="ruleset file to fit an objective for")
    fit.add_argument("--store", default="fitted", help="directory of fitted objectives")
    fit.add_argument("--workers", type=int, default=0)
    redo = fit.add_mutually_exclusive_group()
    redo.add_argument(
        "--force", action="store_true", help="refit even if a fit is already stored"
    )
    redo.add_argument(
        "--check",
        action="store_true",
        help="re-derive a fit made on another core: record the loaded core beside the "
             "store if it reproduces, save the new fit if it does not",
    )
    fit.set_defaults(handler=_cmd_fit)

    values = sub.add_parser(
        "values",
        help="search this ruleset's piece values, and check them against the derived table",
    )
    values.add_argument("ruleset", help="ruleset file to search a valuation for")
    values.add_argument(
        "--store", default="valuations", help="directory of searched valuations"
    )
    values.add_argument("--workers", type=int, default=0)
    values.add_argument(
        "--quiet", action="store_true", help="do not print every comparison"
    )
    values.add_argument(
        "--force", action="store_true", help="search again even if one is already stored"
    )
    values.set_defaults(handler=_cmd_values)

    _add_train(sub)

    mutants = sub.add_parser(
        "mutants", help="write the calibration mutant corpus as ruleset files"
    )
    mutants.add_argument("--out", default="runs/mutants", help="directory to write into")
    mutants.set_defaults(handler=_cmd_mutants)

    labelled = sub.add_parser(
        "calibration-corpus",
        help="write the labelled corpus calibrate and anchor read: every committed game "
             "and every mutant, played or adopted, and analysed",
    )
    labelled.add_argument("out", type=Path, help="directory to write, e.g. runs/v11")
    labelled.add_argument("--rulesets", type=Path, default=corpus.RULESET_DIR)
    labelled.add_argument(
        "--library",
        type=Path,
        default=RUNS_DIR,
        help="the runs root whose current seed-0 runs are adopted rather than played; "
             "read only",
    )
    labelled.add_argument(
        "--mutant",
        action="append",
        default=[],
        metavar="NAME",
        help="only this mutant (repeatable); the default is every one that builds",
    )
    labelled.add_argument(
        "--score",
        default=None,
        metavar="WEIGHTS",
        help="also score every entry under this weights version; nothing is scored "
             "without it",
    )
    labelled.add_argument("--workers", type=int, default=0)
    labelled.set_defaults(handler=_cmd_calibration_corpus)

    calibrate = sub.add_parser(
        "calibrate", help="score an index against a labelled corpus of mutants"
    )
    calibrate.add_argument("root", help="directory holding the scored run directories")
    calibrate.add_argument("--weights", default=None)
    calibrate.add_argument("--json", action="store_true")
    calibrate.set_defaults(handler=_cmd_calibrate)

    anchor = sub.add_parser(
        "anchor",
        help="report where the band sits against a weights version's pinned curve ends "
             "(writes nothing; since v12 the band checks the scale and places none of it)",
    )
    anchor.add_argument("root", help="directory holding the scored run directories, or runs/")
    anchor.add_argument("--weights", default=DEFAULT_WEIGHTS)
    anchor.add_argument("--json", action="store_true")
    anchor.set_defaults(handler=_cmd_anchor)

    verdict = sub.add_parser(
        "verdict", help="ask a model to write, in prose, whether a game is worth playing"
    )
    verdict.add_argument("run", type=Path, help="run directory containing games.jsonl")
    verdict.add_argument("--ruleset", type=Path, default=None)
    verdict.add_argument("--sample", type=int, default=JUDGE_SAMPLE, help="games to show")
    verdict.set_defaults(handler=_cmd_verdict)

    agents = sub.add_parser("agents", help="list the registered agent kinds")
    agents.set_defaults(handler=_cmd_agents)

    fairy = sub.add_parser(
        "fairy",
        help="check which rulesets Fairy-Stockfish can play, and how faithfully",
    )
    fairy.add_argument(
        "rulesets",
        type=Path,
        nargs="*",
        help="rulesets to check; the whole committed corpus when none are named",
    )
    fairy.add_argument("--depth", type=int, default=3, help="how deep to compare the trees")
    fairy.add_argument("--games", type=int, default=3, help="random games to walk")
    fairy.add_argument(
        "--plies",
        type=int,
        default=400,
        help=(
            "plies per walked game (Q438's shogi promotion out of the zone had a median "
            "first difference at ply 108; a 400-ply walk catches it in 38 of 40 random "
            "games against 4 of 40 at 60, Q442)"
        ),
    )
    fairy.add_argument(
        "--result-games",
        type=int,
        default=6,
        help="games played to the end for the result check, which uses pyffish",
    )
    fairy.add_argument(
        "--no-results",
        action="store_true",
        help="check the moves only, not who wins",
    )
    fairy.add_argument(
        "--config",
        action="store_true",
        help="print the generated variant definition instead of checking it",
    )
    fairy.set_defaults(handler=_cmd_fairy)

    smoke = sub.add_parser("smoke", help="run the cheap degeneracy gate on a ruleset")
    smoke.add_argument("ruleset", type=Path)
    smoke.add_argument("--games", type=int, default=150)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--workers", type=int, default=0)
    smoke.set_defaults(handler=_cmd_smoke)

    evolve = sub.add_parser("evolve", help="run the automatic design loop")
    evolve.add_argument("--generations", type=int, default=8)
    evolve.add_argument("--agents", nargs="*", default=None)
    evolve.add_argument("--games", type=int, default=3, help="games per ordered pairing")
    evolve.add_argument("--workers", type=int, default=0)
    evolve.add_argument("--seed", type=int, default=0)
    evolve.add_argument("--weights", default=DEFAULT_WEIGHTS)
    evolve.add_argument("--no-judge", action="store_true", help="score on measurements alone")
    evolve.add_argument(
        "--mechanical-rate", type=float, default=0.5,
        help="share of generations made by a local mechanical edit rather than a model "
             "call; 1.0 runs the whole loop without contacting a model at all",
    )
    evolve.add_argument(
        "--from", dest="seeds", nargs="*", type=Path, default=None,
        help="rulesets to start from; each is measured on this run's roster first",
    )
    evolve.add_argument(
        "--batch", type=int, default=8,
        help="children generated per generation; all are gated, one gets a tournament",
    )
    evolve.add_argument(
        "--promote", type=int, default=1,
        help="how many survivors of the batch reach the rated roster",
    )
    evolve.add_argument("--out", type=Path, default=RUNS_DIR / "design")
    evolve.set_defaults(handler=_cmd_evolve)

    serve = sub.add_parser(
        "serve", help="run the local application: browse, play, and start design runs"
    )
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--rulesets", type=Path, default=Path("rulesets"))
    serve.add_argument("--runs", type=Path, default=RUNS_DIR)
    serve.add_argument(
        "--heuristics",
        type=Path,
        default=Path("heuristics"),
        help="store of heuristic players, offered as opponents per ruleset",
    )
    serve.add_argument(
        "--archive",
        type=Path,
        default=corpus.ARCHIVE_DIR,
        help="games kept out of the corpus; playable, and not calibrated against",
    )
    serve.add_argument("--open", action="store_true", help="open a browser window")
    serve.set_defaults(handler=_cmd_serve)

    site = sub.add_parser(
        "site", help="the public site: the local application with a shorter route table"
    )
    # A group rather than a command, because the site grows commands beside `serve` that
    # are nothing to do with serving -- `load` is next.
    site_sub = site.add_subparsers(dest="site_command", required=True)
    site_serve = site_sub.add_parser(
        "serve",
        help="run the site on localhost; it refuses to start without a database URL "
             "in the environment",
    )
    site_serve.add_argument("--port", type=int, default=8766)
    site_serve.add_argument("--host", default="127.0.0.1")
    site_serve.add_argument("--rulesets", type=Path, default=Path("rulesets"))
    site_serve.add_argument("--runs", type=Path, default=RUNS_DIR)
    site_serve.add_argument("--heuristics", type=Path, default=Path("heuristics"))
    site_serve.add_argument("--archive", type=Path, default=corpus.ARCHIVE_DIR)
    site_serve.set_defaults(handler=_cmd_site_serve)

    site_migrate = site_sub.add_parser(
        "migrate",
        help="apply the numbered SQL files the site's store is made of, in order",
    )
    site_migrate.add_argument(
        "--dry-run",
        action="store_true",
        help="say which migrations would run and write nothing, the ledger included",
    )
    site_migrate.set_defaults(handler=_cmd_site_migrate)

    site_load = site_sub.add_parser(
        "load",
        help="read the ruleset library, the measurement summaries, the picker ladders "
             "and the local application's games into the store; idempotent by hash",
    )
    site_load.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="the checkout to read rulesets/, archive/ and runs/ from (default: here)",
    )
    # The default is `load.MAINTAINER`, resolved in the handler for the same reason
    # `--idle` is: building the parser must not import the site.
    site_load.add_argument(
        "--maintainer",
        default=None,
        help="handle the corpus is owned by and the local application's games sit under "
             "(default: maintainer)",
    )
    site_load.add_argument(
        "--dry-run",
        action="store_true",
        help="say what would change and roll it back rather than committing",
    )
    site_load.set_defaults(handler=_cmd_site_load)

    site_invite = site_sub.add_parser(
        "invite",
        help="invite one address: a code, a link, and the mail that carries it",
    )
    site_invite.add_argument("email", help="the address the invitation is sent to")
    site_invite.add_argument(
        "--issuer",
        default=None,
        help="the handle the invitation is issued by (default: whoever owns the corpus)",
    )
    site_invite.add_argument(
        "--uses",
        type=int,
        default=1,
        help="how many accounts the code may be spent on (default: 1)",
    )
    site_invite.add_argument(
        "--days",
        type=float,
        default=None,
        help="how long the invitation lasts before it expires (default: 14)",
    )
    site_invite.add_argument(
        "--url",
        default=None,
        help="the site's address from the outside (default: CHESS_VARIANT_SANDBOX_SITE_URL)",
    )
    site_invite.add_argument(
        "--no-send",
        action="store_true",
        help="write the invitation and print the link rather than mailing it",
    )
    site_invite.set_defaults(handler=_cmd_site_invite)

    site_revoke = site_sub.add_parser(
        "revoke",
        help="sign everybody out: delete every session row, keeping every account",
    )
    site_revoke.set_defaults(handler=_cmd_site_revoke)

    site_admin = site_sub.add_parser(
        "admin",
        help="make somebody an admin, which is the only way the maintenance page opens",
    )
    site_admin.add_argument("handle", help="who to promote")
    site_admin.add_argument(
        "--remove",
        action="store_true",
        help="put them back to an ordinary member instead",
    )
    site_admin.set_defaults(handler=_cmd_site_admin)

    site_forget = site_sub.add_parser(
        "forget",
        help="what the privacy notice promises leaving does: the address and the "
             "password hash go, the games stay",
    )
    site_forget.add_argument("handle", help="who is leaving")
    site_forget.add_argument(
        "--yes",
        action="store_true",
        help="do it without asking; there is no undo",
    )
    site_forget.set_defaults(handler=_cmd_site_forget)

    site_sweep = site_sub.add_parser(
        "sweep",
        help="resign leaderboard matches nobody has moved in, and rate them as losses; "
             "rest the trial games nobody plays; close and open the month's round",
    )
    # The default is `quota.IDLE_LIMIT`, resolved in the handler rather than here:
    # building the parser must not import the site, which would pull the engine in
    # behind it and make `chess-variant-sandbox --help` pay for a command it is not running.
    site_sweep.add_argument(
        "--idle",
        type=float,
        default=None,
        help="minutes without a move before a match is resigned (default: 30)",
    )
    site_sweep.set_defaults(handler=_cmd_site_sweep)

    site_usage = site_sub.add_parser(
        "usage",
        help="how the site has been used, by the hour and naming nobody; reads the "
             "database, so it is the maintainer's",
    )
    site_usage.add_argument(
        "--days", type=float, default=7.0, help="how far back to read (default: 7)"
    )
    site_usage.add_argument(
        "--route", default=None, help="one route template, as 'POST /api/design/play'"
    )
    site_usage.add_argument(
        "--samples", action="store_true", help="list every five-minute sample as well"
    )
    site_usage.add_argument("--json", action="store_true")
    site_usage.set_defaults(handler=_cmd_site_usage)

    site_games = site_sub.add_parser(
        "games",
        help="every game a person finished on the site, one JSON line each with its "
             "moves, so a win can be replayed; reads the database, so it is the maintainer's",
    )
    site_games.add_argument("--game", default="", help="a game's hash, or a prefix of it")
    site_games.add_argument(
        "--player", default=None, help="one signed-in person's games, by handle"
    )
    site_games.add_argument("--mode", choices=("sandbox", "leaderboard"), default=None)
    site_games.add_argument(
        "--won", action="store_true", help="only the games the person won"
    )
    site_games.set_defaults(handler=_cmd_site_games)

    humans = sub.add_parser(
        "humans",
        help="what happened when a person played: results, and where they sit on the ladder",
    )
    humans.add_argument("--runs", default="runs", help="where the record lives")
    humans.add_argument("--ruleset", default="", help="one ruleset hash, rather than all")
    humans.add_argument("--json", action="store_true")
    humans.set_defaults(handler=_cmd_humans)

    board = sub.add_parser("leaderboard", help="show the state of a design run")
    board.add_argument("--out", type=Path, default=RUNS_DIR / "design")
    board.set_defaults(handler=_cmd_leaderboard)

    # The three things you can do with a game the loop invented, in the order of how
    # much they commit you. See `chess_variant_sandbox/corpus.py` for why they are three and not one.
    test = sub.add_parser(
        "test", help="measure a candidate on the roster a committed score means"
    )
    test.add_argument("hash", help="content hash of the game, from `chess-variant-sandbox leaderboard`")
    test.add_argument("--runs", type=Path, default=RUNS_DIR)
    test.add_argument("--weights", default=DEFAULT_WEIGHTS)
    test.add_argument("--workers", type=int, default=0)
    # Four runs, because this is the rung before `promote` and a committed score is the
    # mean of four. `measure --seeds` is where a person chooses otherwise.
    test.set_defaults(handler=_cmd_test, seeds="four", adopt=None)

    measure = sub.add_parser(
        "measure",
        help="bring a game's canonical-roster score up to date, or check what is stale",
    )
    measure.add_argument(
        "hash",
        nargs="?",
        default="",
        help="content hash or ruleset path; omit for every committed and archived game",
    )
    measure.add_argument("--runs", type=Path, default=RUNS_DIR)
    measure.add_argument("--weights", default=DEFAULT_WEIGHTS)
    measure.add_argument("--workers", type=int, default=0)
    measure.add_argument(
        "--check",
        action="store_true",
        help="report what is out of date and exit non-zero, measuring nothing",
    )
    measure.add_argument(
        "--force", action="store_true", help="replay even what is already current"
    )
    measure.add_argument(
        "--seeds",
        choices=("one", "four"),
        default="",
        help="how many runs this game's score is the mean of; the default is what its "
             "rung expects -- one for a candidate, the library's set for a committed or "
             "archived game",
    )
    measure.add_argument(
        "--adopt",
        type=Path,
        default=None,
        metavar="ROOT",
        help="a campaign root of <hash>-<seed> directories to take runs from instead of "
             "playing them; a run is taken only where a sample of it re-derives here",
    )
    measure.set_defaults(handler=_cmd_measure)

    refresh = sub.add_parser(
        "refresh",
        help="keep the library on the current scorer: repair what is stale, one game at a "
             "time, from the refresher's own worktree",
    )
    refresh.add_argument(
        "hash", nargs="?", default="",
        help="repair this one game here and exit (what the refresher runs for each game)",
    )
    refresh.add_argument(
        "--library", action="store_true",
        help="the refresher: every stale library game, cheapest repair first, one lease each",
    )
    refresh.add_argument("--watch", action="store_true",
                         help="with --library, keep going: sleep when nothing is owed")
    refresh.add_argument("--no-slow-tier", action="store_true",
                         help="with --watch, skip the weekly `pytest -m slow` run")
    refresh.add_argument("--plan", action="store_true",
                         help="print the queue, in order, and run nothing")
    refresh.add_argument("--status", action="store_true",
                         help="print what the refresher is doing, from its status file")
    refresh.add_argument("--forget", default="", metavar="HASH",
                         help="clear a refused game so the refresher takes it again")
    refresh.add_argument("--only", action="append", default=[], metavar="GAME",
                         help="with --library, only this game (a hash or a ruleset path); "
                              "repeatable")
    refresh.add_argument("--json", action="store_true", help="machine-readable output")
    refresh.add_argument("--runs", type=Path, default=None,
                         help="the runs/ to read and deliver into; default the main checkout's")
    # No default here: the refresher's children read their own commit's `DEFAULT_WEIGHTS`,
    # so a weights bump on main is the refresher's next revision rather than pinned out.
    refresh.add_argument("--weights", default=None,
                         help="score under this weights version instead of the current one")
    refresh.add_argument("--workers", type=int, default=0)
    refresh.add_argument("--interval", type=float, default=600.0,
                         help="seconds between looks when nothing is owed or the machine is busy")
    refresh.add_argument("--wait", type=int, default=3600,
                         help="seconds to wait for each game's lease before looking again")
    refresh.add_argument("--worktree", type=Path, default=None,
                         help="the refresher's worktree; default .claude/worktrees/refresher "
                              "under the main checkout")
    refresh.set_defaults(handler=_cmd_refresh)

    strength = sub.add_parser(
        "premise",
        help="check whether a game's players are strong enough for its score (Q502d)",
    )
    strength.add_argument(
        "hash",
        nargs="?",
        default="",
        help="content hash or ruleset path; omit for every committed and archived game",
    )
    strength.add_argument("--runs", type=Path, default=RUNS_DIR)
    strength.add_argument("--workers", type=int, default=0)
    strength.add_argument(
        "--check",
        action="store_true",
        help="report each game's verdict and what is unchecked or stale, playing nothing",
    )
    strength.add_argument(
        "--force", action="store_true", help="check again even where the verdict is current"
    )
    strength.set_defaults(handler=_cmd_premise)

    keep = sub.add_parser(
        "archive", help="keep a candidate in the repository, out of the corpus"
    )
    keep.add_argument("hash")
    keep.add_argument("--note", default="", help="why it is worth keeping")
    keep.add_argument(
        "--credit", default="",
        help="the designer's handle: written as provenance.author and into the history",
    )
    keep.add_argument("--name", default="", help="rename the game as it is kept")
    keep.add_argument("--runs", type=Path, default=RUNS_DIR)
    keep.add_argument("--out", type=Path, default=corpus.ARCHIVE_DIR)
    keep.set_defaults(handler=_cmd_archive)

    admit = sub.add_parser("promote", help="admit a candidate to the calibration corpus")
    admit.add_argument("hash")
    admit.add_argument(
        "--reason",
        default="",
        help="what this game pins down that nothing already in the corpus does",
    )
    admit.add_argument("--name", default="", help="rename the game as it is admitted")
    admit.add_argument("--runs", type=Path, default=RUNS_DIR)
    admit.add_argument("--out", type=Path, default=corpus.RULESET_DIR)
    admit.set_defaults(handler=_cmd_promote)

    viz = sub.add_parser("viz", help="render a game to a self-contained HTML page")
    viz.add_argument("run", type=Path, help="a run directory or a games.jsonl file")
    viz.add_argument("--ruleset", type=Path, default=None)
    viz.add_argument("--game", type=int, default=0, help="index of the game within the corpus")
    viz.add_argument("--out", type=Path, default=None)
    viz.add_argument(
        "--letters",
        action="store_true",
        help="draw spec letters instead of piece icons",
    )
    viz.set_defaults(handler=_cmd_viz)

    accuracy = sub.add_parser(
        "accuracy",
        help="judge one finished game move by move, through winning chances fitted on its run",
    )
    accuracy.add_argument("run", type=Path, help="a run directory or a games.jsonl file")
    accuracy.add_argument("--ruleset", type=Path, default=None)
    accuracy.add_argument("--game", type=int, default=0, help="index of the game to judge")
    accuracy.add_argument("--seed", type=int, default=0, help="seeds the held-out split")
    accuracy.add_argument(
        "--positions-per-game",
        type=int,
        default=None,
        help="read at most this many positions of each game for the fit (default: all)",
    )
    accuracy.add_argument("--out", type=Path, default=None, help="also write the report as JSON")
    accuracy.set_defaults(handler=_cmd_accuracy)

    return parser


# --------------------------------------------------------------------- commands


def _cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.ruleset)
    errors = validate_spec(spec)
    if errors:
        print(f"{args.ruleset}: {len(errors)} problem(s)")
        for error in errors:
            print(f"  - {error}")
        return 1

    engine = Engine(spec)
    complexity = measure_complexity(spec)
    state = engine.initial_state()
    print(f"{spec['name']} [{spec_hash(spec)}] is valid")
    print(f"  board       {engine.spec.width}x{engine.spec.height}")
    print(f"  pieces      {', '.join(engine.spec.piece_ids)}")
    print(f"  opening     {len(engine.legal_moves(state))} legal moves")
    print(f"  rule mass   {complexity.rule_mass:.1f}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    """Throughput on the two operations that dominate a sweep.

    Any change to the engine gets a before/after from this command. It defaults to one
    thread on purpose: the number has to stay comparable across machines and across
    the Python-to-Rust move, and a core count is not part of the engine.
    """
    spec = load_spec(args.ruleset)
    engine = Engine(spec)
    state = engine.initial_state()

    started = time.perf_counter()
    nodes = engine.perft(state, args.depth, threads=args.threads)
    perft_seconds = time.perf_counter() - started

    started = time.perf_counter()
    playouts = engine.core.random_playouts(args.games, 0, float("inf"), args.threads)
    playout_seconds = time.perf_counter() - started
    plies = sum(game[0] for game in playouts)

    print(f"{spec['name']} [{spec_hash(spec)}]  ({args.threads} thread(s))")
    print(f"  perft({args.depth})      {nodes:,} leaves in {perft_seconds:.2f}s "
          f"({nodes / max(perft_seconds, 1e-9):,.0f} leaves/s)")
    print(f"  random playouts  {len(playouts)} games, {plies:,} plies in {playout_seconds:.2f}s "
          f"({len(playouts) / max(playout_seconds, 1e-9):,.1f} games/s)")
    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    spec = load_spec(args.ruleset)
    engine = Engine(spec)
    first, second = args.agents
    record = play_game(engine, (first, second), args.seed)
    if not args.quiet:
        state = engine.initial_state()
        for index, move in enumerate(record.moves):
            marker = "." if index % 2 == 0 else "..."
            print(f"{index // 2 + 1}{marker} {engine.describe_move(state, move)}")
            state = engine.apply(state, move)

    winner = "draw" if record.winner is None else f"player {record.winner} ({record.agents[record.winner]})"
    print(f"\n{winner} by {record.reason} in {record.plies} plies ({record.duration_ms:.0f} ms)")
    return 0


def _cmd_tournament(args: argparse.Namespace) -> int:
    spec = load_spec(args.ruleset)
    config = TournamentConfig(
        agents=tuple(args.agents),
        games_per_pairing=args.games,
        seed=args.seed,
        workers=args.workers,
    )
    total = len(config.agents) ** 2 * config.games_per_pairing
    print(f"{spec['name']} [{spec_hash(spec)}]: {total} games across {len(config.agents)} agents")

    started = time.perf_counter()
    # Q732: resumable under --checkpoint; without it, the runner's own run_tournament.
    from chess_variant_sandbox.tournament.checkpoint import run_tournament

    result = run_tournament(spec, config, progress=True)
    elapsed = time.perf_counter() - started

    out = args.out or _tournament_dir(result.ruleset_hash, args.seed)
    result.save(out)

    print(f"\n{len(result.records)} games in {elapsed:.1f}s -> {out}")
    _print_standings(result.ratings)
    return 0


def _tournament_dir(digest: str, seed: int) -> Path:
    """Where a tournament given no `--out` goes: `runs/tournament/<hash>-<seed>`.

    Not `corpus.run_dir`, which is the library's. The tournament plays whatever roster it
    is asked for, and `corpus._foreign_run` only guards `measure`'s side: it refuses to
    replay over somebody's one-off, but nothing stopped the one-off landing on a library
    run in the first place. A caller that means to write a library run goes through
    `corpus.measure`, which plays the canonical roster and stamps the file that says so.
    """
    return TOURNAMENT_DIR / f"{digest}-{seed}"


def _cmd_score(args: argparse.Namespace) -> int:
    run = Path(args.run)
    games_path = run / "games.jsonl" if run.is_dir() else run
    records = list(read_records(games_path))
    if not records:
        raise ValueError(f"no games found in {games_path}")

    spec = load_spec(args.ruleset) if args.ruleset else _find_ruleset(records[0].ruleset_hash)

    # The second pass is optional and its absence is not an error: a corpus measured
    # before it existed is still a valid corpus. Weights that ask for a component it
    # supplies will name the gap rather than scoring a missing number as zero.
    analysis = None
    if run.is_dir() and (run / "analysis.json").exists():
        from chess_variant_sandbox.tournament.analysis import CorpusAnalysis

        analysis = CorpusAnalysis.load(run / "analysis.json")

    # The third pass, likewise optional -- and absent for a stronger reason, since it
    # needs a committed heuristic store rather than only a replay.
    heuristics = None
    if run.is_dir() and (run / "heuristics.json").exists():
        from chess_variant_sandbox.tournament.heuristics import HeuristicReport

        heuristics = HeuristicReport.load(run / "heuristics.json")

    result = score_ruleset(
        spec,
        records,
        rate(records),
        weights=args.weights,
        analysis=analysis,
        heuristics=heuristics,
        interval=not args.no_interval,
    )

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(result.summary())
        print("\n  component                score   weight-share   raw")
        # ``contributions`` is empty whenever there is no total to contribute to: a
        # corpus missing a measurement, or a ruleset this roster cannot read. The
        # components that *were* measured are still worth printing -- the next question
        # is always "how badly?" -- so the sort falls back to the score.
        for name in sorted(
            result.scores, key=lambda n: -result.contributions.get(n, result.scores[n])
        ):
            share = result.contributions.get(name)
            printed = "--" if share is None else f"{share:.3f}"
            print(
                f"  {name:<24} {result.scores[name]:>5.2f}   "
                f"{printed:>10}   {result.raw[name]:.3f}"
            )
        if result.unreadable:
            print(
                "\n  withheld, because this roster cannot read this ruleset: "
                + ", ".join(result.unreadable)
            )
        _print_depth_reached(result.diagnostics.get("depth_by_agent") or {})

    if run.is_dir():
        result.save(run / f"index-{result.weights_version}.json")
    return 0 if result.viable else 1


def _print_depth_reached(depths: dict[str, Any]) -> None:
    """How deep each searching rung got, beside the numbers that read the ladder.

    Printed with the score because it is the context those numbers are missing: a
    ruleset whose top rung never completed two plies is a ruleset whose skill numbers
    describe the budget. Nothing is gated on it -- see stage 27 in ``docs/plan.md`` and
    the node-budget confound in ``docs/bugs.md``.

    Absent from a corpus recorded before the quantity was carried, which is not an
    error: the field defaults, so an old run simply has nothing to say here.
    """
    if not depths:
        return
    print("\n  how deep each rung saw, in completed plies per search")
    for name, entry in sorted(depths.items(), key=lambda item: -item[1]["mean_depth"]):
        print(f"  {name:<24} {entry['mean_depth']:>5.2f}   over {entry['searches']} searches")


def _cmd_verdict(args: argparse.Namespace) -> int:
    """Write ``<run>/verdict.json``, which the library shows next to the rules.

    A separate command rather than part of ``score`` because it costs a model call and
    a score does not, and because the two are different kinds of answer: one says how
    this game measured, the other says what it is and whether anyone would want it.
    Nothing downstream may turn the second into the first -- there is no number in it.
    """
    from chess_variant_sandbox.llm import judge_ruleset

    run = Path(args.run)
    games_path = run / "games.jsonl" if run.is_dir() else run
    records = list(read_records(games_path))
    if not records:
        raise ValueError(f"no games found in {games_path}")

    spec = load_spec(args.ruleset) if args.ruleset else _find_ruleset(records[0].ruleset_hash)
    judgement = judge_ruleset(spec, records, sample=args.sample)

    payload = {"ruleset_hash": spec_hash(spec), **judgement.as_dict()}
    print(f"{spec['name']} [{payload['ruleset_hash']}]\n")
    print(judgement.verdict)
    for label, items in (("strengths", judgement.strengths), ("weaknesses", judgement.weaknesses)):
        for item in items:
            print(f"  {label[0]}: {item}")
    if judgement.suspected_degeneracy:
        print(f"\n  suspected degeneracy: {judgement.suspected_degeneracy}")

    if run.is_dir():
        path = run / "verdict.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


def _cmd_agents(args: argparse.Namespace) -> int:
    print("registered agent kinds:")
    for kind in available_agents():
        print(f"  {kind}")
    print("\nagents take options as kind:key=value,key=value")
    print("budget is always in nodes, so the same option means the same work in any ruleset")
    return 0


def _cmd_fairy(args: argparse.Namespace) -> int:
    """Report which rulesets the outside engine can play, and why not for the rest.

    Three outcomes and they are different news. *Refused* means there is no
    translation and the reasons say which rule has no counterpart -- that is a fact
    about the two rule systems and no engine will change it. *Diverged* means there is
    a translation and the two engines disagree about a position, which is a defect
    here or a limit of the installed build, and the position is printed so it can be
    looked at. *Verified* is the only one that makes the opponent available.

    Verified is about the moves. Under each translation a second line says whether the
    two engines also agree who has won (`fairy.check_results`, Q404). That check needs
    `pyffish`, because the UCI binary cannot be asked whether a game is over, so it
    reads the library's build rather than the binary's, and the two verdicts are kept
    apart: *move-verified* against the installed engine, *result-verified* against
    `pyffish`.
    """
    from chess_variant_sandbox import fairy
    from chess_variant_sandbox.corpus import RULESET_DIR

    paths = list(args.rulesets) or sorted(RULESET_DIR.glob("*.json"))
    if args.config:
        for path in paths:
            variant = fairy.translation(load_spec(path))
            print(f"# {path}")
            print(variant.config if variant else "# no translation")
        return 0

    binary = fairy.binary_path()
    if binary is None:
        print(
            "no Fairy-Stockfish binary found. Put one on PATH or set "
            f"{fairy.BINARY_ENV}; nothing is downloaded for you. "
            "Only the translation can be checked without one."
        )
    else:
        print(f"engine: {binary}")
    referee, library = (None, "") if args.no_results else _fairy_referee()
    if referee is not None:
        print(f"results: {library}")
    elif not args.no_results:
        print(f"results: not checked; {library}")

    verified = refused = diverged = 0
    agree = differ = both = 0
    for path in paths:
        spec = load_spec(path)
        name = spec.get("name", str(path))
        reasons = fairy.why_not(spec)
        if reasons:
            refused += 1
            print(f"  refused  {name:24s} {'; '.join(reasons)}")
            continue
        moves_ok = False
        if binary is None:
            print(f"  writes   {name:24s} translated; no engine to check it against")
        else:
            variant = fairy.translation(spec)
            assert variant is not None
            with fairy.UciEngine() as engine:
                result = fairy.verify(
                    spec,
                    engine,
                    fairy.config_path(variant),
                    depth=args.depth,
                    games=args.games,
                    plies=args.plies,
                )
            moves_ok = result.ok
            if result.ok:
                verified += 1
                print(f"  verified {name:24s} {result.summary()}")
            else:
                diverged += 1
                print(f"  DIVERGED {name:24s} {result.summary()}")
                if result.fen:
                    print(f"           {'':24s} {result.fen}")
        if referee is None:
            continue
        check = fairy.check_results(spec, referee, games=args.result_games)
        if check.ok:
            agree += 1
            both += moves_ok
            print(f"           {'results':24s} {check.summary()}")
            continue
        differ += 1
        print(f"           {'RESULTS DIFFER':24s} {check.coverage()}")
        for difference in check.differences:
            print(f"           {'':24s} {difference.text()}")
            print(f"           {'':24s}   {difference.fen}")
        for reason in check.reasons:
            print(f"           {'':24s} {reason}")
    print(
        f"\n{verified} verified, {diverged} diverged, {refused} refused, "
        f"of {len(paths)} rulesets"
    )
    if referee is not None:
        print(
            f"results against {library}: {agree} agree, {differ} differ, "
            f"of {agree + differ} translations"
        )
        if binary is not None:
            print(f"{both} both move-verified and result-verified")
    return 1 if diverged or differ else 0


def _fairy_referee() -> tuple[Any, str]:
    """`pyffish` as the result check's referee and its version, or ``None`` and why not."""
    try:
        import pyffish  # noqa: PLC0415 - optional, and only where it is used

        from chess_variant_sandbox.fairy.verify import PyffishBackend
    except ImportError:
        return None, "pyffish is not installed (it is a dev dependency)"
    version = ".".join(str(part) for part in pyffish.version())
    return PyffishBackend(), f"pyffish {version}"


def _cmd_smoke(args: argparse.Namespace) -> int:
    from chess_variant_sandbox.loop import smoke_test

    spec = load_spec(args.ruleset)
    result = smoke_test(spec, games=args.games, seed=args.seed, workers=args.workers)
    print(f"{spec['name']} [{spec_hash(spec)}]: {'passed' if result.passed else 'REJECTED'}")
    for reason in result.reasons:
        print(f"  - {reason}")
    for key, value in sorted(result.stats.items()):
        print(f"  {key:<16} {value}")
    return 0 if result.passed else 1


def _cmd_evolve(args: argparse.Namespace) -> int:
    from chess_variant_sandbox.loop import LoopConfig, evolve

    config = LoopConfig(
        generations=args.generations,
        agents=tuple(args.agents) if args.agents else LoopConfig.agents,
        games_per_pairing=args.games,
        workers=args.workers,
        seed=args.seed,
        weights=args.weights,
        judge=not args.no_judge,
        mechanical_rate=args.mechanical_rate,
        batch=args.batch,
        promote=args.promote,
        seeds=tuple(args.seeds) if args.seeds else (),
        directory=args.out,
    )
    evolve(config)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Serve the local application.

    Localhost by design: sessions live in memory, a design run spends money, and nothing
    here authenticates anyone. It is a tool for the person at the keyboard.
    """
    from chess_variant_sandbox.web import serve

    serve(
        host=args.host,
        port=args.port,
        ruleset_dir=args.rulesets,
        runs_dir=args.runs,
        archive_dir=args.archive,
        open_browser=args.open,
        heuristics_dir=args.heuristics,
    )
    return 0


def _cmd_site_serve(args: argparse.Namespace) -> int:
    """Serve the public site: the same app, registering only the routes §2 allows.

    Localhost like `serve`, and for the opposite reason -- not because nobody else should
    reach it, but because everyone else reaches it through Caddy, which holds the TLS and
    the rate limit.
    """
    from chess_variant_sandbox.site import serve

    serve(
        host=args.host,
        port=args.port,
        ruleset_dir=args.rulesets,
        runs_dir=args.runs,
        archive_dir=args.archive,
        heuristics_dir=args.heuristics,
    )
    return 0


def _cmd_site_migrate(args: argparse.Namespace) -> int:
    """Bring the site's store up to the schema the repository describes.

    Run on every deploy, before the site starts. It is safe to run on a database that is
    already current, which is the point -- the deploy script does not have to know.
    """
    from chess_variant_sandbox.site import db, migrate

    url = db.database_url()
    print(f"migrating {db.redacted(url)}")
    with db.connect(url) as connection:
        print(migrate.run(connection, dry_run=args.dry_run).summary())
    return 0


def _cmd_site_load(args: argparse.Namespace) -> int:
    """Put the repository's own data in the store, and say what moved.

    Run on every deploy, after the migration and before the restart. It is safe on a
    store that is already loaded, which is the point: the second run writes nothing and
    says so.
    """
    from chess_variant_sandbox.site import db, load

    url = db.database_url()
    print(f"loading {args.root} into {db.redacted(url)}")
    with db.connect(url) as connection:
        report = load.load(connection, args.root, args.maintainer or load.MAINTAINER)
        if args.dry_run:
            connection.rollback()
        else:
            connection.commit()
    print(report.summary())
    if args.dry_run:
        print("dry run: nothing was committed")
    return 0


def _cmd_site_invite(args: argparse.Namespace) -> int:
    """Invite one person, by email. The only way an account comes into being.

    The row and the mail are one act: the invitation is written, the message goes, and
    only then does the transaction commit -- so a send that fails leaves no code that
    nobody was told about. `--no-send` is the other half of that honesty, for a box with
    no key on it and for reading the link out over something else.
    """
    from datetime import timedelta

    from chess_variant_sandbox.site import db, identity, mail

    lifetime = (
        identity.INVITE_LIFETIME if args.days is None else timedelta(days=args.days)
    )
    with db.connect(db.database_url()) as connection:
        issuer = _site_issuer(connection, args.issuer)
        invite, where = identity.send_invitation(
            connection,
            args.email,
            args.url or mail.site_url(),
            None if args.no_send else mail.send,
            issuer=issuer,
            uses=args.uses,
            lifetime=lifetime,
        )
        if args.no_send:
            print(f"invited {invite.email}, not sent. The link is:\n  {where}")
        else:
            print(f"invited {invite.email}: the link is on its way")
        connection.commit()
    days = lifetime.total_seconds() / 86400
    print(f"good for {invite.uses} account(s), expires in {days:g} days")
    return 0


def _site_issuer(connection: Any, handle: str | None) -> int | None:
    """Who the invitation is from.

    A handle asked for has to exist. Asked for nothing, this is the maintainer the loader
    made -- and `None` on a store that has never been loaded, which is the first invite
    the schema already allows for.
    """
    from chess_variant_sandbox.site import load

    wanted = handle or load.MAINTAINER
    with connection.cursor() as cursor:
        cursor.execute("select id from users where lower(handle) = lower(%s)", (wanted,))
        found = cursor.fetchone()
    if found is None and handle:
        raise ValueError(f"no user with the handle {handle!r}")
    return None if found is None else int(found[0])


def _cmd_site_revoke(args: argparse.Namespace) -> int:
    """Sign everybody out at once -- §8's answer to unexpected traffic.

    Sessions only. Accounts, games and ratings stay, and everybody can sign in again with
    the password they already have; a person who should not come back at all is 1.8's
    per-user switch.
    """
    from chess_variant_sandbox.site import db, identity

    with db.connect(db.database_url()) as connection:
        gone = identity.revoke_sessions(connection)
        connection.commit()
    print(f"revoked {gone} session(s); every account is intact")
    return 0


def _cmd_site_admin(args: argparse.Namespace) -> int:
    """Promote somebody to admin, or put them back. The only way `/admin` opens.

    There is no button anywhere that makes another admin, because the page is reached by
    being one and a promotion loop with no floor is how one mistaken account becomes
    every account. So it is this command, run against the database by whoever has the
    box.
    """
    from chess_variant_sandbox.site import admin, db

    role = admin.MEMBER if args.remove else admin.ADMIN
    with db.connect(db.database_url()) as connection:
        handle = admin.set_role(connection, args.handle, role)
        connection.commit()
    if role == admin.ADMIN:
        print(f"{handle} is an admin; the maintenance page is at /admin")
    else:
        print(f"{handle} is an ordinary member again")
    return 0


def _cmd_site_forget(args: argparse.Namespace) -> int:
    """Leaving, as `/privacy` describes it and no further.

    The email address and the password hash are deleted, every session ends, and the name
    on every match becomes *a former player*. The matches stay: each game's ladder is
    replayed from its first match, so removing rows would move other people's ratings
    along with this one's. Nothing here can be undone, which is why it asks.
    """
    from chess_variant_sandbox.site import admin, db

    if not args.yes:
        answer = input(
            f"Forget {args.handle}? Their address and password hash are deleted and "
            f"every match becomes 'a former player'. This cannot be undone. [y/N] "
        )
        if answer.strip().lower() not in ("y", "yes"):
            print("nothing was changed")
            return 0
    with db.connect(db.database_url()) as connection:
        handle = admin.forget(connection, args.handle)
        connection.commit()
    print(f"{handle} has been forgotten; their games stay and read 'a former player'")
    return 0


def _cmd_site_usage(args: argparse.Namespace) -> int:
    """The usage readings the admin page shows, as text or JSON (`docs/plan-usage.md` §4.3).

    It reads the database directly, so whoever can run it already holds the store.
    """
    from chess_variant_sandbox.site import db, usage

    since, until = usage.period(args.days)
    with db.connect(db.database_url()) as connection:
        found = usage.report(connection, since, until, args.route)
        box = usage.samples(connection, since, until)
    if args.json:
        payload: dict[str, object] = {"report": found.as_dict(), "box": box.as_dict()}
        if args.samples:
            payload["samples"] = [usage.sample_dict(row) for row in box.rows]
        print(json.dumps(payload, indent=2))
    else:
        print(usage.render(found, box, listed=args.samples))
    return 0


def _cmd_site_games(args: argparse.Namespace) -> int:
    """The finished human games the store kept, as JSON lines (Q502a).

    One line per finish, oldest first, with the moves as the engine's integers, so a
    reader replays one with `site.finishes.replay` against the game's spec. No line names
    a person: `--player` narrows by handle and the lines it prints still carry none.
    Like `site usage` it reads the database directly, so it runs where the store is.
    """
    from chess_variant_sandbox.site import db, finishes

    with db.connect(db.database_url()) as connection:
        found = finishes.recorded(
            connection, game=args.game, handle=args.player, mode=args.mode, won=args.won
        )
    for finish in found:
        print(json.dumps(finish.as_dict(), separators=(",", ":")))
    return 0


def _cmd_site_sweep(args: argparse.Namespace) -> int:
    """Resign the leaderboard matches nobody came back to, rest the trial games nobody
    plays, and close and open the month's round.

    §4 of the plan: leaving a leaderboard game counts as a resignation. A tab closed
    leaves no signal at all, so the limit is what decides, and the resignation is
    recorded as a finish -- which is what makes the rating update run on it. Nothing is
    ever deleted. Run from the deploy's timer, or by hand.

    The resting sweep (Q503i, `site/resting.py`) and the round's (Q503h,
    `site/rounds.py`) are the same kind of clock and run here too, as `site load` runs
    them; a second run the same day does nothing.
    """
    from datetime import timedelta

    from chess_variant_sandbox.site import db, matches, quota, resting, rounds

    idle = quota.IDLE_LIMIT if args.idle is None else timedelta(minutes=args.idle)
    url = db.database_url()
    with db.connect(url) as connection:
        resigned = matches.sweep(connection, idle)
        rested = resting.rest(connection)
        swept = rounds.sweep(connection)
        connection.commit()
    minutes = idle.total_seconds() / 60
    print(f"idle past {minutes:g} minutes: {len(resigned)} match(es) resigned")
    for found in resigned:
        print(f"  {found.id}  {found.game[:12]}  {found.plies} plies")
    print(f"resting: {len(rested)} trial game(s) rested")
    for rest in rested:
        print(f"  {rest.said}")
    for line in swept.lines:
        print(f"round: {line}")
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    """Measure one candidate on the roster a committed score means, on four seeds.

    Kept as the name the archive and the web interface already say, and a thin spelling
    of `measure --force --seeds four`: "test this candidate" has always meant "play it on
    the corpus roster whatever is on disk", which is exactly the forced case, and this is
    the rung before `promote`, where a committed score is the mean of four runs.
    """
    return _measure_one(args.hash, args, force=True)


def _cmd_measure(args: argparse.Namespace) -> int:
    """Bring measurements up to date, doing only the work that is actually missing.

    The front door, and the reason the other three commands no longer have to be
    remembered in the right order. `tournament`, `analyse` and `score` still exist and
    still do what they say -- they are the right tools for a one-off run on a roster of
    your choosing -- but the sequence that produces a number comparable with `chess` is
    this one, and getting it wrong used to leave a directory that looked measured and
    was not.
    """
    targets = _measure_targets(args)
    if not targets:
        print("nothing to measure")
        return 0

    if args.check:
        return _report_staleness(targets, args)

    return _measure_all(targets, args)


def _measure_all(targets: list[str], args: argparse.Namespace) -> int:
    """Repair every target, the re-score-only games side by side (R7 of the perf survey).

    The repairs are planned first. A game that owes a tournament, a fit or anything
    `corpus.poolable` cannot vouch for runs here, in target order, as it always did: it
    already uses every core. The games that owe only a re-score or a re-analysis of games
    on disk then go through a process pool of `--workers` processes, each doing exactly
    what `_measure_one` does, and what each said is printed whole, in target order, so
    the log reads as the serial one would. `--workers 1` is the serial path.

    Nothing about a game's answer depends on which path it took: each writes only its own
    directories, from its own records, with no seed drawn from anything shared, and
    `tests/test_corpus.py` holds the two paths' files byte for byte.
    """
    import os

    pool_size = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    seeds = _seed_set(args)
    pooled: dict[str, corpus.Candidate] = {}
    if pool_size > 1 and not args.force and getattr(args, "adopt", None) is None:
        for digest in targets:
            try:
                candidate = corpus.find(digest, runs_dir=args.runs)
            except (ValueError, OSError):
                continue
            if corpus.poolable(candidate, args.runs, args.weights, seeds):
                pooled[digest] = candidate
    if len(pooled) < 2:
        pooled = {}

    failed = 0
    for digest in targets:
        if digest in pooled:
            continue
        try:
            _measure_one(digest, args, force=args.force)
        except (ValueError, OSError) as error:
            failed += 1
            print(f"  {digest}: {error}")

    if pooled:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # `spawn` everywhere, as Windows has no other: a worker imports the package from
        # disk and inherits nothing of this process but its arguments, so the path is the
        # same on every platform.
        with ProcessPoolExecutor(
            max_workers=min(pool_size, len(pooled)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = [
                pool.submit(
                    _measure_pooled, candidate, args.runs, args.weights, seeds
                )
                for candidate in pooled.values()
            ]
            for future in futures:
                lines, refusal = future.result()
                for line in lines:
                    print(line)
                if refusal:
                    failed += 1
                    print(refusal)
    return 1 if failed else 0


def _measure_pooled(
    candidate: corpus.Candidate,
    runs: Path,
    weights: str,
    seeds: tuple[int, ...] | None,
) -> tuple[list[str], str]:
    """One pooled game, as `_measure_one` would repair it, with what it said kept.

    The analysis pass, where a re-analysis owes one, runs on one core: the pool is the
    parallelism, and a pass per worker spreading over every core would oversubscribe the
    machine by the pool's size. Threads never change an answer, so it is the same pass.
    """
    lines: list[str] = []
    try:
        _measure_candidate(
            candidate, runs, weights, workers=1, force=False, seeds=seeds,
            adopt=None, say=lines.append,
        )
    except (ValueError, OSError) as error:
        return lines, f"  {candidate.hash}: {error}"
    return lines, ""


def _measure_targets(args: argparse.Namespace) -> list[str]:
    """Which games this invocation is about: the named one, or everything committed."""
    if args.hash:
        # A path is accepted as well as a hash, because what a person has in hand after
        # editing a ruleset is the file, and making them look its hash up first is the
        # sort of friction this command exists to remove.
        path = Path(args.hash)
        if path.exists():
            return [spec_hash(load_spec(path))]
        return [args.hash]

    return corpus.corpus_hashes()


def _report_staleness(targets: list[str], args: argparse.Namespace) -> int:
    """Say what is out of date and what it would cost, without spending it.

    Exits non-zero when anything is stale, so it works as a check in a script or a hook.
    "The scores on disk no longer describe the code in this tree" is the sort of thing
    worth failing a build over, and it was not previously a question anything could ask.
    """
    statuses = corpus.survey(targets, runs_dir=args.runs, weights=args.weights)
    stale = 0
    for status in statuses:
        if not status.current:
            stale += 1
        # The run count travels with the word `current`, because a mean of four runs and
        # a single run are not the same claim. A stale line says the reason instead, and
        # `Staleness.cost` already counts the runs inside it.
        said = status.reason or f"current, {status.runs} run{'' if status.runs == 1 else 's'}"
        print(f"  {status.name:22} [{status.hash}] {said}")

    behind = stale_build()
    if behind is not None:
        print(f"\nand the core itself: {behind}")
    print(f"\n{stale} of {len(targets)} need work")
    return 1 if stale else 0


def _cmd_refresh(args: argparse.Namespace) -> int:
    """The refresher (EV5, `docs/plan-evolving-score.md` §6); see `refresher.py`."""
    from chess_variant_sandbox import refresher

    if args.hash and (args.library or args.plan or args.status or args.forget):
        # The lease hook reads these flags as the supervisor's (no lease); a named game is
        # a repair in this tree and needs one, so the two never share a line.
        print("a named game is repaired here; --library, --plan, --status and --forget "
              "name no game", file=sys.stderr)
        return 2
    if args.hash:
        runs = args.runs or RUNS_DIR
        outcome = refresher.repair(
            args.hash, runs, args.weights, workers=args.workers,
            say=None if args.json else print,
        )
        if args.json:
            print(json.dumps(outcome.as_dict()))
        elif outcome.refused:
            print(f"refused ({outcome.refusal}): {outcome.reason}")
        return 1 if outcome.refused else 0

    if args.plan:
        items = refresher.queue(args.runs or RUNS_DIR, args.weights)
        if args.json:
            print(json.dumps([item.as_dict() for item in items]))
            return 0
        for item in items:
            print(f"  {item.needs:9} tier {item.tier}  {item.name:22} [{item.hash}] "
                  f"{item.lease}: {item.reason}")
        print(f"\n{len(items)} owed")
        return 0

    main = refresher.main_checkout_of(Path.cwd())
    runs = args.runs or main / "runs"
    if args.status:
        status = refresher.read_status(runs)
        print(json.dumps(status, indent=2) if args.json else
              "\n".join(refresher.plain_status(status)))
        return 0
    if args.forget:
        found = refresher.Ledger(runs).forget(args.forget)
        print(f"{args.forget}: {'cleared' if found else 'was not refused'}")
        return 0
    if not args.library:
        print("name a game, or pass --library (the refresher), --plan or --status",
              file=sys.stderr)
        return 2

    worktree = (args.worktree or main / refresher.WORKTREE).resolve()
    if worktree == main.resolve():
        print("the refresher never measures from the main checkout", file=sys.stderr)
        return 2
    machine = refresher.GitMachine(
        main_checkout=main, worktree=worktree, weights=args.weights or "",
        workers=args.workers, wait_s=args.wait, runs_dir=runs,
    )
    if not machine.exists():
        print(f"making the refresher's worktree at {worktree}, and building its core")
        machine.create()
        if not machine.acquire("build", "refresh worktree"):
            print("no build lease for the refresher's first build", file=sys.stderr)
            return 1
        try:
            failed = machine.rebuild()
        finally:
            machine.release()
        if failed:
            print(failed, file=sys.stderr)
            return 1
    only = frozenset(
        spec_hash(load_spec(Path(game))) if Path(game).exists() else game for game in args.only
    ) or None
    loop = refresher.Refresher(
        machine, refresher.Ledger(runs), watch=args.watch, interval=args.interval,
        say=print, only=only, slow_tier=False if args.no_slow_tier else None,
    )
    return loop.run()


def _cmd_premise(args: argparse.Namespace) -> int:
    """Check whether each game's players are strong enough for its score (Q502d).

    Beside the score, never in it: the verdict goes to `runs/<hash>/premise.json` with its
    own basis, and no score moves, and nothing reads it as a gate: a failing verdict marks
    the score on the page and holds nothing out of the band (withdrawn 2026-10-01, Q645).
    `--check` plays nothing and exits non-zero while any
    target is unchecked or stale, as `measure --check` does.
    """
    from chess_variant_sandbox import premise

    targets = _measure_targets(args)
    owed = 0
    failed = 0
    for digest in targets:
        try:
            candidate = corpus.find(digest, runs_dir=args.runs)
        except (ValueError, OSError) as error:
            failed += 1
            print(f"  {digest}: {error}")
            continue
        found = premise.read(digest, args.runs, candidate.spec)
        if args.check or (found is not None and found.current and not args.force):
            if found is None:
                said = "not checked yet"
            elif found.current:
                said = found.recorded
            else:
                said = "stale: " + "; ".join(found.reasons) + (
                    " (last verdict failed)" if found.marked else ""
                )
            if found is None or not found.current:
                owed += 1
            print(f"  {candidate.name:22} [{digest}] {said}")
            continue
        behind = stale_build()
        if behind is not None:
            print(behind)
            return 1
        started = time.monotonic()
        try:
            payload = premise.check(candidate.spec, workers=args.workers)
        except (ValueError, OSError) as error:
            failed += 1
            print(f"  {candidate.name:22} [{digest}] {error}")
            continue
        premise.write(payload, args.runs)
        ceiling = payload["ceiling"]
        fairy = payload["fairy"]
        against = (
            f"; {fairy['score']:.3f} +/- {fairy['error']:.3f} against {fairy['reference']}"
            if fairy.get("played") else f"; Fairy-Stockfish not played: {fairy['why']}"
        )
        print(
            f"  {candidate.name:22} [{digest}] {payload['verdict']}: "
            f"{ceiling['score']:.3f} +/- {ceiling['error']:.3f} against "
            f"{ceiling['reference']}{against} ({time.monotonic() - started:.0f} s)",
            flush=True,
        )
    if args.check:
        print(f"\n{owed} of {len(targets)} unchecked or stale")
        return 1 if owed else 0
    return 1 if failed else 0


def _seed_set(args: argparse.Namespace) -> tuple[int, ...] | None:
    """Which seeds this invocation asks for, or ``None`` for whatever the rung expects.

    Two words rather than a list of numbers. Which seeds a run count *is* belongs to
    `corpus`, and a person choosing between one run and four is choosing between a
    visitor's score and a library one, not between arbitrary integers.
    """
    return {"one": corpus.ONE_SEED, "four": corpus.LIBRARY_SEEDS}.get(
        getattr(args, "seeds", "") or ""
    )


def _measure_one(digest: str, args: argparse.Namespace, force: bool) -> int:
    candidate = corpus.find(digest, runs_dir=args.runs)
    _measure_candidate(
        candidate, args.runs, args.weights, workers=args.workers, force=force,
        seeds=_seed_set(args), adopt=getattr(args, "adopt", None), say=print,
    )
    return 0


def _measure_candidate(
    candidate: corpus.Candidate,
    runs: Path,
    weights: str,
    workers: int,
    force: bool,
    seeds: tuple[int, ...] | None,
    adopt: Path | None,
    say: Any,
) -> None:
    """`corpus.refresh` on one game, with its report, whichever path runs it."""
    say(f"{candidate.name} [{candidate.hash}] -- {candidate.stage}")
    if candidate.run is not None:
        say(f"  from {candidate.run}")
    stored = corpus.refresh(
        candidate, runs_dir=runs, weights=weights, workers=workers,
        report=lambda line: say(f"  {line}"), force=force,
        seeds=seeds, adopt_from=adopt,
    )
    say(f"  {corpus.measurement_dir(candidate.hash, runs)}")
    if not stored.stored.get("viable"):
        say("  not viable on this roster; it cannot be promoted as it stands")


def _cmd_archive(args: argparse.Namespace) -> int:
    """Keep a candidate in the repository without admitting it to the corpus."""
    candidate = corpus.find(args.hash, runs_dir=args.runs)
    path = corpus.archive_game(
        candidate, note=args.note, archive_dir=args.out, runs_dir=args.runs,
        name=args.name, credit=args.credit,
    )
    print(f"archived {candidate.name} [{candidate.hash}] -> {path}")
    print("  it stays playable and is not part of the calibration corpus")
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    """Admit a candidate to the calibration corpus, with the argument for doing so."""
    candidate = corpus.find(args.hash, runs_dir=args.runs)
    path = corpus.promote(
        candidate, reason=args.reason, ruleset_dir=args.out, runs_dir=args.runs,
        name=args.name,
    )
    print(f"promoted {candidate.name} [{candidate.hash}] -> {path}")
    print(
        "  it is now part of what the index is calibrated against: give it a row in\n"
        "  `.claude/rules/rulesets-corpus.md` saying what it pins down, and expect the\n"
        "  corpus-wide tests to have an opinion"
    )
    return 0


def _cmd_humans(args: argparse.Namespace) -> int:
    """Report the only ground truth the project has, and refuse to flatter it.

    Every other measurement here compares the ladder against itself. This one places a
    person on it -- against opponent ratings that were computed without them and are
    held fixed, so a result cannot move the ladder it is being read against. Nothing
    downstream reads this: it is an anchor and a sanity check, never a target.
    """
    from chess_variant_sandbox.web.human import HumanRecord, picker_ratings
    from chess_variant_sandbox.web.opponents import stored_path

    runs = Path(args.runs)
    record = HumanRecord(runs)
    games = record.games(args.ruleset)
    if not games:
        print(
            f"no human games recorded under {record.path}.\n"
            f"Play some: uv run chess-variant-sandbox serve --open"
        )
        return 1

    summaries = []
    for digest in dict.fromkeys(game.ruleset_hash for game in games):
        table = None
        path = stored_path(digest, runs)
        if path.exists():
            table = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(record.summary(digest, picker_ratings(table)))

    if args.json:
        print(json.dumps(summaries, indent=2))
        return 0

    print(f"{len(games)} sessions across {len(summaries)} rulesets, from {record.path}\n")
    print(f"  {'ruleset':<24}{'played':>7}{'left':>6}{'score':>8}{'you':>9}"
          f"{'top rung':>10}{'gap':>8}")
    for entry in sorted(summaries, key=lambda e: -e["finished"]):
        anchor = entry["anchor"]
        score = "--" if entry["score"] is None else f"{entry['score']:.3f}"
        if anchor is None:
            rating = gap = top = "--"
        else:
            # A sweep is a bound, so it is printed as one: `>` means "somewhere
            # above this ladder", `<` means below, and a bare number means placed.
            edge = {"up": ">", "down": "<"}.get(anchor["swept"], "")
            rating = f"{edge}{anchor['rating']:.0f}"
            top = f"{anchor['top_rating']:.0f}"
            gap = f"{anchor['top_above_human']:+.0f}"
        print(
            f"  {entry['name'][:23]:<24}{entry['finished']:>7}{entry['abandoned']:>6}"
            f"{score:>8}{rating:>9}{top:>10}{gap:>8}"
        )
    print(
        "\n  'left' is games begun and never finished, which is the bias automatic\n"
        "  recording cannot remove; read it before the scores. 'gap' is how far the\n"
        "  strongest opponent offered sits above the person, in Elo. A ruleset with no\n"
        "  anchor has no rated picker yet -- rate one from the library page.\n"
        "  A 'you' of >n or <n is a sweep: the games bound the person above or\n"
        "  below the ladder rather than placing them on it."
    )
    for entry in sorted(summaries, key=lambda e: -e["finished"]):
        if not entry["byOpponent"]:
            continue
        print(f"\n  {entry['name']}")
        for row in entry["byOpponent"]:
            elo = "--" if row["rating"] is None else f"{row['rating']:.0f}"
            print(
                f"    {row['agent'][:38]:<39}{row['games']:>3} games  "
                f"you scored {row['score']:.3f}   their elo {elo}"
            )
    return 0


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    """The state of a design run: the grid first, the ranked board second.

    That order is the point. A ranked board answers "what is the best game found",
    whose answer sits inside its own error bars; the grid answers "what kinds of game
    have been found, and what is the best of each", which is a question this search can
    actually settle.
    """
    from chess_variant_sandbox.loop import Archive
    from chess_variant_sandbox.loop.preference import PreferenceLog

    archive = Archive(args.out)
    if not archive.entries:
        print(f"no design run found under {args.out}")
        return 1

    table = PreferenceLog(args.out).table()
    grid = archive.grid(prefer=table)
    if grid.filled:
        print(grid.summary())
        print()
    # The current revision's board (EV3): a row scored under another is placed, marked
    # and counted in the header, and never compared with a current one.
    print(archive.summary(revision=DEFAULT_WEIGHTS))
    if table.comparisons:
        print(
            f"\n  judge: {table.comparisons} comparisons, reversed on "
            f"{table.flip_rate:.0%} when the pair was swapped"
        )
    return 0


def _cmd_viz(args: argparse.Namespace) -> int:
    from chess_variant_sandbox.viz import render_game

    run = Path(args.run)
    games_path = run / "games.jsonl" if run.is_dir() else run
    records = list(read_records(games_path))
    if not records:
        raise ValueError(f"no games found in {games_path}")
    if not 0 <= args.game < len(records):
        raise ValueError(f"game index {args.game} outside 0..{len(records) - 1}")

    record = records[args.game]
    spec = load_spec(args.ruleset) if args.ruleset else _find_ruleset(record.ruleset_hash)
    out = args.out or (run if run.is_dir() else run.parent) / f"game-{args.game}.html"
    path = render_game(spec, record, out, icons=not args.letters)
    print(f"wrote {path}")
    return 0


def _cmd_accuracy(args: argparse.Namespace) -> int:
    """Fit winning chances on a run's games, then judge one of them move by move.

    Reads only: no record, score, rating or verdict is written or moved
    (`docs/reference/win-probability.md`). ``--out`` is the one file it writes.
    """
    from chess_variant_sandbox.accuracy import judge, move_texts, render
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.winprob import fit_run, read_game

    run = Path(args.run)
    games_path = run / "games.jsonl" if run.is_dir() else run
    records = list(read_records(games_path))
    if not records:
        raise ValueError(f"no games found in {games_path}")
    if not 0 <= args.game < len(records):
        raise ValueError(f"game index {args.game} outside 0..{len(records) - 1}")

    record = records[args.game]
    spec = load_spec(args.ruleset) if args.ruleset else _find_ruleset(record.ruleset_hash)
    engine = Engine(spec)
    win = fit_run(engine, records, seed=args.seed, positions_per_game=args.positions_per_game)
    reading = read_game(engine, record, win.ruler, index=args.game)
    report = judge(reading, win, move_texts(engine, record, reading))
    print(render(report, spec.get("name", "")))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.as_dict(), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


# ---------------------------------------------------------------------- helpers


def _find_ruleset(digest: str) -> dict[str, Any]:
    """Locate the spec a record names, by content hash.

    Records store the hash rather than a path, so a corpus stays interpretable after
    files move. The search covers the committed rulesets and anything the design loop
    has archived.
    """
    for directory in (Path("rulesets"), RUNS_DIR / "archive"):
        if not directory.exists():
            continue
        for path in directory.rglob("*.json"):
            try:
                candidate = load_spec(path)
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(candidate, dict) and spec_hash(candidate) == digest:
                return candidate
    raise FileNotFoundError(
        f"no ruleset with hash {digest} under rulesets/ or runs/archive/; "
        "pass --ruleset to point at it directly"
    )


def _print_standings(ratings: RatingTable) -> None:
    print(f"\n  {'agent':<26} {'elo':>7} {'games':>6} {'score':>7} {'draws':>6} {'as first':>9}")
    for standing in ratings.standings:
        print(
            f"  {standing.name:<26} {standing.rating:>7.0f} {standing.games:>6} "
            f"{standing.score_rate:>7.3f} {standing.draws:>6} {standing.first_player_score:>9.3f}"
        )


def _cmd_analyse(args: argparse.Namespace) -> int:
    """Run the engine-derived second pass over a finished run.

    Separate from ``tournament`` because it needs no new games: a record stores moves,
    and the engine re-derives every position from them. An existing corpus gains the
    position-derived components without replaying anything.
    """
    from chess_variant_sandbox.tournament.analysis import analyse_corpus

    run = Path(args.run)
    games_path = run / "games.jsonl" if run.is_dir() else run
    records = list(read_records(games_path))
    if not records:
        raise ValueError(f"no games found in {games_path}")

    spec = load_spec(args.ruleset) if args.ruleset else _find_ruleset(records[0].ruleset_hash)
    started = time.perf_counter()
    analysis = analyse_corpus(
        spec, records, blunder_sample=args.blunders, workers=args.workers, progress=True
    )
    elapsed = time.perf_counter() - started

    target = run if run.is_dir() else run.parent
    path = analysis.save(target / "analysis.json")
    samples = sum(len(g.near_best) for g in analysis.games)
    print(
        f"{spec['name']}: {len(analysis.games)} games analysed, {samples} positions "
        f"scored, {len(analysis.blunders)} blunder trials in {elapsed:.1f}s"
    )
    print(f"wrote {path}")
    return 0


def _cmd_heuristics(args: argparse.Namespace) -> int:
    """Generate a ruleset's heuristic players, then measure what depth buys over them.

    Two steps in one command because they are two halves of one idea, and separable by
    ``--generate-only`` because the first is the expensive, non-deterministic half. A
    set is written once and committed; the measurement reads it and never re-samples.
    """
    from chess_variant_sandbox.agents.heuristics import HeuristicSet, load_set, store_path
    from chess_variant_sandbox.agents.seeds import handwritten_set, random_set
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.tournament.heuristics import (
        heuristic_components,
        heuristic_diagnostics,
        measure_heuristics,
    )

    spec = load_spec(args.ruleset)
    engine = Engine(spec)
    compiled = engine.spec

    existing = load_set(compiled.hash, args.store)
    if existing is not None and args.source == "handwritten":
        heuristic_set: HeuristicSet = existing
        print(f"{compiled.name}: reusing {store_path(compiled.hash, args.store)}")
    else:
        if args.source == "model":
            from chess_variant_sandbox.llm.heuristics import propose_heuristics

            heuristic_set = propose_heuristics(spec, compiled, k=max(1, args.k - 1))
        elif args.source == "random":
            heuristic_set = random_set(compiled, k=args.k, seed=args.seed)
        else:
            heuristic_set = handwritten_set(compiled, k=args.k)
        path = heuristic_set.save(store_path(compiled.hash, args.store))
        print(f"{compiled.name}: wrote {heuristic_set.k} heuristics to {path}")

    heuristic_set.register()
    if args.generate_only:
        for heuristic in heuristic_set.heuristics:
            print(f"  {heuristic.digest}  {heuristic.name:<24} {heuristic.plan}")
        return 0

    baseline = ""
    if args.against_fit:
        from chess_variant_sandbox.agents.fitted import load_fit, procedure_digest

        baseline = procedure_digest()
        if load_fit(compiled.hash, baseline) is None:
            raise FileNotFoundError(
                f"no fitted objective for {compiled.name} under procedure {baseline}; "
                f"run `chess-variant-sandbox fit {args.ruleset}` first"
            )

    started = time.perf_counter()
    report = measure_heuristics(
        spec,
        heuristic_set,
        seed=args.seed,
        games=args.games,
        nodes=args.nodes or None,
        workers=args.workers,
        baseline_fit=baseline,
    )
    elapsed = time.perf_counter() - started

    out = Path(args.out) if args.out else Path("runs") / f"{compiled.hash}-heuristics.json"
    report.save(out)

    print(f"{compiled.name}: {elapsed:.1f}s at {report.nodes} nodes -> {out}")
    print(f"  depth reached {report.depth_reached}  measurable={report.measurable}")
    print()
    print(f"  {'plan':<26}{'headroom':>10}{'legible':>9}{'agrees':>8}")
    for row in report.rows:
        print(
            f"  {row.name:<26}{row.headroom:>10.3f}{row.legibility:>9.3f}"
            f"{row.agreement:>8.3f}"
        )
    print()
    for name, value in heuristic_components(report).items():
        print(f"  {name:<28}{value}")
    diagnostics = heuristic_diagnostics(report)
    correlation = diagnostics.get("headroom_legibility_correlation")
    if correlation is not None:
        print(f"  {'headroom~legibility':<28}{correlation}")
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    """Fit a weight vector for this ruleset, then ask whether it earned its place.

    The check is the command's point, not a postscript. Stage 23 showed that *proposing*
    weights over this vocabulary does not work; fitting them might, and the only thing
    that separates the two claims is a match against the objective being replaced. A
    refused fit is still written out, because a failure that leaves no file is a failure
    nobody can read.

    ``--check`` asks whether a stored fit is still the loaded core's answer (Q80b,
    `docs/reference/refit-repair.md`). It exits 0 whatever the verdict, since the question
    it answers is reproduction and it leaves the store answering it either way.
    """
    from chess_variant_sandbox.agents.features import VOCABULARY
    from chess_variant_sandbox.agents.fitted import (
        FittedObjective,
        procedure_digest,
        stale_verdict,
        store_path,
    )
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.measurement import built_fingerprint
    from chess_variant_sandbox.tournament.fitting import check_fit, fit_ruleset, verified_cores

    spec = load_spec(args.ruleset)
    compiled = Engine(spec).spec
    procedure = procedure_digest()
    path = store_path(compiled.hash, procedure, args.store)

    if args.check:
        started = time.perf_counter()
        found = check_fit(spec, store=args.store, workers=args.workers)
        fit = found.fit
        print(f"{compiled.name}: {_reproduction(found, time.perf_counter() - started)}")
    elif path.exists() and not args.force:
        fit = FittedObjective.load(path)
        print(f"{compiled.name}: reusing {path}")
    else:
        started = time.perf_counter()
        fit = fit_ruleset(spec, workers=args.workers)
        fit.save(path)
        print(f"{compiled.name}: fitted in {time.perf_counter() - started:.1f}s -> {path}")

    verdict = "accepted" if fit.accepted else "REFUSED, falls back to the built-in objective"
    print(f"  procedure {procedure}  {fit.samples} positions  loss {fit.loss:.4f}")
    print(
        f"  against the built-in objective at equal budget: "
        f"{fit.accept_score:.4f} +/-{fit.accept_error:.3f} over {fit.accept_games} "
        f"games -- {verdict}"
    )
    # Reported, never enforced, exactly as `values` reports it. A verdict taken on
    # another core is still a real number about a real player; what it is no longer is a
    # number about *this* engine, and a reader has to be told which they are looking at.
    # A store a check has reproduced on this core is this core's answer, whoever made it.
    reproduced = built_fingerprint() in verified_cores(fit, args.store)
    drifted = None if reproduced else stale_verdict(fit)
    if drifted:
        print(f"  NOTE: {drifted}")
    for term, weight in zip(VOCABULARY, fit.vector, strict=True):
        if weight:
            print(f"    {term:<22} {weight:+.4f}")
    if args.check:
        return 0
    return 0 if fit.accepted else 1


# `train`'s flags, each a field of `training.procedure.Procedure`: the flag's name, its
# type and its help. A flag left out keeps the default procedure's value, so the default
# run is `minimax:net=<DEFAULT.digest>` on every board.
_TRAIN_FLAGS: tuple[tuple[str, type, str], ...] = (
    ("games", int, "self-play games per ordered pairing of the roster"),
    ("seed", int, "the self-play schedule's seed"),
    ("opening_plies", int, "each self-play game opens with 0..n random plies (0: none)"),
    ("skip_plies", int, "opening plies that are never samples"),
    ("stride", int, "sample every n-th ply after the skipped ones"),
    ("result_weight", float, "weight of the game's result in the label; the rest is the probe"),
    ("label_search", str, "the label's search: probe (the analysis probe), player (minimax:nodes=<label-nodes>'s own root value) or parent (the same on the --parent net)"),
    ("label_nodes", int, "the label search's node budget"),
    ("label_depth", int, "the probe's depth cap (the player's search does not read it)"),
    ("td_lambda", float, "TD(lambda) along the game line for the probe's part (default 0.7; 0: the probe alone, the default before Q671)"),
    ("scale", float, "built-in units per logistic unit of the output"),
    ("accumulator", int, "accumulator width A (<= 2048)"),
    ("hidden1", int, "first hidden width (<= 128)"),
    ("hidden2", int, "second hidden width (<= 128)"),
    ("epochs", int, "passes over the positions"),
    ("batch", int, "positions per optimiser step"),
    ("learning_rate", float, "Adam's step size"),
    ("init_seed", int, "the seed of the initial weights and the shuffles"),
)


def _add_train(sub: Any) -> None:
    train = sub.add_parser(
        "train",
        help="train this ruleset's learned evaluator from self-play, duel it, and store it",
    )
    train.add_argument("ruleset", help="ruleset file to train a net for")
    train.add_argument("--store", default="nets", help="directory of net stores")
    train.add_argument("--workers", type=int, default=0, help="threads and processes (0: all but one)")
    train.add_argument(
        "--agents", nargs="+", default=None, help="the self-play roster (default: the procedure's)"
    )
    for name, kind, text in _TRAIN_FLAGS:
        train.add_argument("--" + name.replace("_", "-"), dest=name, type=kind, default=None, help=text)
    train.add_argument(
        "--duel-pairs", type=int, default=None, help="paired seeds in the acceptance duel, each from its own random opening (default 200)"
    )
    train.add_argument(
        "--duel-nodes", type=int, default=None, help="both players' node budget in the duel (default 6400)"
    )
    train.add_argument(
        "--no-duel", action="store_true",
        help="skip the duel: the store's verdict is 'unchecked', and it never plays",
    )
    train.add_argument(
        "--curtail", action="store_true",
        help="stop the acceptance duel once its verdict is settled (plan-compute §4.2): the "
        "same verdict, but the stored score and error are the prefix's, which plan-teacher "
        "§3's gates read as measurements; so off by default",
    )
    train.add_argument("--save-data", default=None, help="also write the harvested positions here (.npz)")
    train.add_argument(
        "--harvest-only", action="store_true",
        help="play and harvest, write --save-data, and stop: no training, duel or store",
    )
    train.add_argument(
        "--data", default=None, help="train from positions saved by --save-data, not new self-play"
    )
    train.add_argument(
        "--force", action="store_true", help="train again even if this procedure's store exists"
    )
    train.add_argument(
        "--parent", default=None,
        help="a generation (plan-teacher §9.1): the parent net's <ruleset-hash>-<procedure>.net "
        "(or its .json); the recipe becomes the generation's, the flags still apply",
    )
    train.add_argument(
        "--replay", default=None, metavar="PARENT_JSON",
        help="with --parent: also train on the parent's own training games (plan-teacher "
        "§9.13.3, E1's replay set), read from the parent's store JSON's recipe; the replay "
        "roster's net must resolve on this ruleset (an accepted store under nets/, or --replay-net)",
    )
    train.add_argument(
        "--replay-net", default=None,
        help="with --replay: the replay roster's net's <ruleset-hash>-<procedure>.net, registered "
        "in this process as the parent is (a refused net, e.g. a guard's generation 0)",
    )
    train.add_argument(
        "--also-harvest", nargs="+", choices=("C", "K"), default=(),
        help="with --harvest-only: also harvest §9.1's control C (the default's probe) and/or "
        "arm K (the built-in player at --label-nodes) from the same games, beside --save-data "
        "as <stem>.C.npz and <stem>.K.npz",
    )
    train.add_argument(
        "--duel-only", action="store_true",
        help="train nothing: duel this procedure's stored net (registered in this process "
        "whatever its verdict) and print the result as one JSON line; nothing is written",
    )
    train.add_argument(
        "--duel-against", default=None,
        help="with --duel-only: the opponent (default minimax:nodes=<duel-nodes>), e.g. the "
        "parent's player or minimax:nodes=25600",
    )
    train.add_argument(
        "--duel-seed", type=int, default=None,
        help="with --duel-only: the duel's first seed (default 1,000,000; 2,000,000 confirms)",
    )
    train.set_defaults(handler=_cmd_train)


def _cmd_train(args: argparse.Namespace) -> int:
    """Train, judge and store one ruleset's net (Q502f, `docs/reference/learned-evaluator.md` §8).

    The store is written whatever the verdict, as `fit` writes a refused fit: a refusal
    that leaves no file is one nobody can read. Only an accepted net plays.
    """
    try:
        from chess_variant_sandbox.agents.training.data import Dataset
        from chess_variant_sandbox.agents.training.procedure import (
            ACCEPT_NODES,
            ACCEPT_PAIRS,
            DEFAULT,
            Procedure,
            generation,
            replayed,
        )
        from chess_variant_sandbox.agents.training.store import (
            read_parent,
            register_parent,
            train_ruleset,
        )
    except ImportError as error:
        print(
            f"train needs numpy ({error}); install the extra: uv sync --extra train",
            file=sys.stderr,
        )
        return 2
    from chess_variant_sandbox.agents.nets import store_paths

    changes: dict[str, Any] = {
        name: getattr(args, name) for name, _kind, _text in _TRAIN_FLAGS
        if getattr(args, name) is not None
    }
    if args.agents:
        changes["agents"] = tuple(args.agents)
    spec = load_spec(args.ruleset)
    compiled = Engine(spec).spec
    base = DEFAULT
    try:
        if args.parent:
            # Q677: a generation. The parent plays in this process (and its harvest
            # workers) under its own name, checked against the bytes the name carries.
            parent, blob = read_parent(args.parent, compiled.hash)
            register_parent(compiled.hash, parent, blob)
            base = generation(parent)
            print(f"{compiled.name}: parent net={parent.procedure} (weights {parent.weights_sha256[:12]})")
        if args.replay:
            # Q713: E1's replay set, the parent's own training games, from its recipe.
            if not args.parent:
                raise ValueError("--replay goes with --parent: a replay set is a generation's")
            stored = json.loads(Path(args.replay).read_text(encoding="utf-8"))
            if "recipe" not in stored:
                raise ValueError(f"{args.replay} holds no recipe to replay")
            base = replayed(base, Procedure.from_dict(stored["recipe"]))
            assert base.replay is not None
            if args.replay_net:
                net, net_blob = read_parent(args.replay_net, compiled.hash)
                if net != base.replay.net:
                    raise ValueError(f"{args.replay_net} is not the replay roster's net "
                                     f"{base.replay.net.procedure} by its bytes")
                register_parent(compiled.hash, net, net_blob)
        elif args.replay_net:
            raise ValueError("--replay-net goes with --replay")
            print(f"  replay set: {base.replay.net.procedure}'s schedule, "
                  f"{base.replay.games} games a pairing, seed {base.replay.seed}")
        procedure = base.with_(**changes)
    except (ValueError, OSError) as error:
        print(f"train: {error}", file=sys.stderr)
        return 2
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    if (args.duel_against or args.duel_seed is not None) and not args.duel_only:
        print("train: --duel-against and --duel-seed go with --duel-only; a stored verdict "
              "is only the acceptance duel", file=sys.stderr)
        return 2
    if args.curtail and (args.no_duel or args.duel_only or args.harvest_only):
        print("train: --curtail curtails the acceptance duel, which this run does not play",
              file=sys.stderr)
        return 2
    if args.also_harvest and not args.harvest_only:
        print("train: --also-harvest goes with --harvest-only", file=sys.stderr)
        return 2
    if args.duel_only:
        return _duel_only(args, spec, compiled.hash, procedure, workers)
    if args.harvest_only:
        # Q658: the positions alone, with their line, for procedures that relabel them
        # (`--data ... --td-lambda L`); the digest says whether they are a stored net's.
        from chess_variant_sandbox.agents.training.data import harvest, self_play

        if not args.save_data:
            print("train: --harvest-only needs --save-data", file=sys.stderr)
            return 2
        if args.also_harvest:
            return _harvest_arms(args, spec, procedure, workers)
        records = self_play(spec, procedure, workers=workers)
        harvested = harvest(spec, records, procedure, workers=workers, report=print)
        print(f"  positions saved to {harvested.save(args.save_data)}")
        print(
            f"  {harvested.positions} positions from {harvested.games} games "
            f"(data {harvested.digest[:12]})"
        )
        return 0
    meta, _weights = store_paths(compiled.hash, procedure.digest, args.store)
    if meta.exists() and not args.force:
        print(f"{compiled.name}: {meta} exists; --force trains it again")
        return 0

    data = Dataset.load(args.data) if args.data else None
    print(f"{compiled.name}: training procedure {procedure.digest}")
    started = time.perf_counter()
    done = train_ruleset(
        spec,
        procedure,
        root=args.store,
        workers=workers,
        pairs=args.duel_pairs if args.duel_pairs is not None else ACCEPT_PAIRS,
        nodes=args.duel_nodes if args.duel_nodes is not None else ACCEPT_NODES,
        duel=not args.no_duel,
        data=data,
        report=print,
        curtail=args.curtail,
    )
    if args.save_data:
        print(f"  positions saved to {done.data.save(args.save_data)}")
    print(f"  {done.data.positions} positions from {done.data.games} games (data {done.data.digest[:12]})")
    print(f"  loss by epoch: {', '.join(f'{loss:.5f}' for loss in done.history)}")
    found = done.acceptance
    if found is None:
        print("  no duel: verdict unchecked, the player is the built-in objective")
    else:
        seconds = done.seconds.get("duel", 0.0)
        rate = (found.candidate_nodes + found.reference_nodes) / seconds if seconds else 0.0
        print(
            f"  {found.candidate} against {found.reference}: {found.score:.4f} +/-"
            f"{found.error:.4f} over {found.games} games -- {done.record.verdict}"
        )
        if found.planned and found.decided_after < found.planned:
            print(f"  curtailed: decided after {found.decided_after} of {found.planned} pairs")
        print(
            f"  nodes: net {found.candidate_nodes}, built-in {found.reference_nodes}; "
            f"{rate:,.0f} nodes/s both seats together ({seconds:.1f}s)"
        )
    print(f"  digest {done.record.digest}, stored {done.meta_path} "
          f"in {time.perf_counter() - started:.1f}s")
    print(f"  plays as minimax:net={procedure.digest},nodes=N where accepted")
    return 0


def _harvest_arms(args: argparse.Namespace, spec: dict, procedure: Any, workers: int) -> int:
    """`train --harvest-only --also-harvest C K`: one self-play, a harvest per arm (§9.1)."""
    from chess_variant_sandbox.agents.training.data import harvest_arms
    from chess_variant_sandbox.agents.training.procedure import builtin_player_arm, control

    makers = {"C": control, "K": builtin_player_arm}
    names = ["child", *args.also_harvest]
    arms = [procedure, *(makers[name](procedure) for name in args.also_harvest)]
    try:
        found = harvest_arms(spec, arms, workers=workers, report=print)
    except ValueError as error:
        print(f"train: {error}", file=sys.stderr)
        return 2
    target = Path(args.save_data)
    for name, arm, data in zip(names, arms, found, strict=True):
        path = target if name == "child" else target.with_name(f"{target.stem}.{name}.npz")
        data.save(path)
        print(
            f"  {name}: procedure {arm.digest}, teacher {data.teacher}, {data.positions} "
            f"positions from {data.games} games (data {data.digest[:12]}) -> {path}"
        )
    return 0


def _duel_only(
    args: argparse.Namespace, spec: dict, ruleset_hash: str, procedure: Any, workers: int
) -> int:
    """`train --duel-only`: one paired duel of a stored net, printed, nothing written.

    The net is the store's weights under `--store`, registered as accepted in this process
    only (a refused or research net duels as itself); the opponent is `--duel-against`,
    by default the built-in at equal nodes (D2). plan-teacher §9.1's P, A25, re-duels and
    confirmations are this command.
    """
    import hashlib

    from chess_variant_sandbox.agents.nets import ACCEPTED, NetRecord, register_net, store_paths
    from chess_variant_sandbox.agents.training.procedure import (
        ACCEPT_NODES,
        ACCEPT_PAIRS,
        ACCEPT_SEED,
    )
    from chess_variant_sandbox.agents.training.store import accept
    from chess_variant_sandbox.measurement import built_fingerprint

    _meta, weights = store_paths(ruleset_hash, procedure.digest, args.store)
    if not weights.exists():
        print(f"train: no net at {weights} to duel", file=sys.stderr)
        return 2
    blob = weights.read_bytes()
    register_net(NetRecord(procedure.digest, ruleset_hash, ACCEPTED, blob))
    started = time.perf_counter()
    try:
        found = accept(
            spec,
            procedure,
            pairs=args.duel_pairs if args.duel_pairs is not None else ACCEPT_PAIRS,
            nodes=args.duel_nodes if args.duel_nodes is not None else ACCEPT_NODES,
            workers=workers,
            seed=args.duel_seed if args.duel_seed is not None else ACCEPT_SEED,
            reference=args.duel_against,
        )
    except ValueError as error:
        print(f"train: {error}", file=sys.stderr)
        return 2
    row = {
        "ruleset_hash": ruleset_hash,
        "procedure": procedure.digest,
        "weights_sha256": hashlib.sha256(blob).hexdigest(),
        "build": built_fingerprint(),
        "duel": found.as_dict(),
        "seconds": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(row), flush=True)
    return 0


def _reproduction(found: Reproduction, seconds: float) -> str:
    """One line for what `fit --check` found and did."""
    from chess_variant_sandbox.tournament import fitting

    made = f"made on core {found.stored.build or 'unknown'}" if found.stored else ""
    if found.outcome == fitting.MADE_HERE:
        return f"{made}, the loaded one; nothing to check -> {found.path}"
    if found.outcome == fitting.REPRODUCED_BEFORE:
        return f"{made}, already reproduced on {found.core} -> {found.sidecar}"
    if found.outcome == fitting.FITTED:
        return f"no stored fit; fitted on core {found.core} in {seconds:.1f}s -> {found.path}"
    if found.outcome == fitting.REFIT:
        return (
            f"{made}, does NOT reproduce on {found.core}: {'; '.join(found.failures)}; "
            f"refit in {seconds:.1f}s -> {found.path}"
        )
    assert found.stored is not None and found.fresh is not None
    score = ""
    if found.fresh.accept_score != found.stored.accept_score:
        score = (
            f"; score {found.stored.accept_score:.4f} -> {found.fresh.accept_score:.4f}, "
            "the same side of the bar"
        )
    return (
        f"{made}, reproduces on {found.core} in {seconds:.1f}s (samples, vector and "
        f"verdict equal{score}) -> {found.sidecar}"
    )


def _cmd_values(args: argparse.Namespace) -> int:
    """Search a piece-value table for this ruleset, then ask whether it earned its place.

    The check is the command's point rather than a postscript, for the reason `fit` says
    it: stage 30B showed that *fitting* piece values to game outcomes does not work, and
    the only thing separating "searched them properly" from "found a different way to
    lose" is a match, on a seed set the search never saw, against the table being
    replaced. A refused search is still written out; a failure that leaves no file is a
    failure nobody can read.
    """
    from chess_variant_sandbox.agents.valuations import (
        SEARCH,
        Valuation,
        stale_verdict,
        store_path,
    )
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.tournament.valuing import timed_search

    spec = load_spec(args.ruleset)
    compiled = Engine(spec).spec
    path = store_path(compiled.hash, SEARCH.digest, args.store)

    if path.exists() and not args.force:
        valuation = Valuation.load(path)
        print(f"{compiled.name}: reusing {path}")
    else:
        print(f"{compiled.name}: searching at {SEARCH.nodes} nodes", flush=True)
        valuation, elapsed = timed_search(
            spec, SEARCH, workers=args.workers, progress=not args.quiet
        )
        valuation.save(path)
        print(f"{compiled.name}: searched in {elapsed:.1f}s -> {path}")

    verdict = "accepted" if valuation.accepted else "REFUSED, falls back to the derived table"
    print(
        f"  procedure {SEARCH.digest}  {valuation.comparisons} comparisons, "
        f"{valuation.accepted_steps} steps kept"
        + ("  (ran out of comparisons)" if valuation.exhausted else "")
    )
    print(
        f"  against the derived table on the same code path: "
        f"{valuation.accept_score:.4f} +/-{valuation.accept_error:.4f} over "
        f"{valuation.accept_games} games -- {verdict}"
    )
    print(
        f"  against minimax:nodes={SEARCH.accept_nodes}: "
        f"{valuation.ladder_score:.4f} +/-{valuation.ladder_error:.4f}"
    )
    # Reported, never enforced. A verdict taken on another core is still a real number
    # about a real player; what it is no longer is a number about *this* engine, and a
    # reader has to be told which they are looking at.
    drifted = stale_verdict(valuation)
    if drifted:
        print(f"  NOTE: {drifted}")
    for piece, value, ratio in zip(
        valuation.piece_ids, valuation.vector, valuation.ratios, strict=True
    ):
        moved = "" if abs(ratio - 1.0) < 1e-9 else f"   x{ratio:.3f}"
        print(f"    {piece:<3} {value:>10.4f}{moved}")
    return 0 if valuation.accepted else 1


def _cmd_mutants(args: argparse.Namespace) -> int:
    """Write the calibration corpus to disk, one spec per mutant."""
    from chess_variant_sandbox.metrics.mutants import build_mutants

    bases: dict[str, Any] = {}
    for path in sorted(Path("rulesets").glob("*.json")):
        spec = load_spec(path)
        bases[spec["name"]] = spec

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    built = build_mutants(bases)
    manifest = []
    for mutant, spec in built:
        errors = validate_spec(spec)
        target = out / f"{mutant.name}.json"
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(spec, handle, indent=1)
        manifest.append(
            {
                "name": mutant.name,
                "base": mutant.base,
                "expectation": mutant.expectation,
                "rationale": mutant.rationale,
                "hash": spec_hash(spec),
                "valid": not errors,
                "errors": errors,
            }
        )
        flag = "" if not errors else f"  INVALID: {errors[0]}"
        print(f"  {mutant.expectation:<6} {mutant.name:<32} {spec_hash(spec)[:12]}{flag}")

    with open(out / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\n{len(built)} mutants written to {out}")
    return 0 if all(m["valid"] for m in manifest) else 1


def _cmd_calibration_corpus(args: argparse.Namespace) -> int:
    """Write the labelled corpus `calibrate` and `anchor` read, with every base named.

    `chess_variant_sandbox/calibration_corpus.py` says what is adopted, what is played and why. Scoring
    is a separate choice, `--score`, because two directories of `runs/` scored under one
    weights version fail the index audit that reads them.
    """
    from chess_variant_sandbox.calibration_corpus import build

    build(
        args.out,
        ruleset_dir=args.rulesets,
        library=args.library,
        only=tuple(args.mutant),
        score=args.score,
        workers=args.workers,
        report=lambda message: print(message, flush=True),
    )
    return 0


# Rulesets that are real entries in the corpus and must be scored, but are not evidence
# about where the human band sits: the two deliberate controls, plus the RPS probes.
# Defined in `metrics/calibration.py` so there is one list rather than two that drift.
from chess_variant_sandbox.metrics.calibration import (  # noqa: E402
    EXCLUDED_FROM_BAND as CONTROL_RULESETS,
)


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Score the index itself against a corpus whose labels are known in advance.

    Reports the four numbers that say whether an index is worth trusting: how far the
    known-bad rulesets sit below the real ones, how many of them land on the right side
    at all, whether a change that alters no gameplay moves the score, and how wide the
    band of real games is -- which should be *narrow*, because they are all good games
    and an index that confidently ranks them is claiming a resolution the sampling
    interval says it does not have.
    """
    from chess_variant_sandbox.metrics.calibration import ScoredEntry, calibrate

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")

    entries: list[ScoredEntry] = []
    # How many runs each entry's index rests on, beside it rather than on it: a
    # `ScoredEntry` is `metrics/calibration.py`'s, in the fingerprinted set.
    runs: list[int] = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        scored = _load_scored(run, args.weights)
        if scored is None:
            continue
        result, label, base = scored
        runs.append(_runs_behind(run, result))
        entries.append(
            ScoredEntry(
                name=result["ruleset_name"],
                total=result["total"],
                viable=result["viable"],
                label=label,
                base=base,
                sd=(result.get("interval") or {}).get("sd", 0.0),
                ladder=result.get("raw", {}).get("ladder_monotonicity", 1.0),
                withheld=bool(result.get("unreadable")),
            )
        )

    if not entries:
        raise ValueError(f"no scored runs under {root}; run 'chess-variant-sandbox score' on them first")

    report = calibrate(entries, weights_version=args.weights or "unspecified")
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.summary())
        print("\n  ruleset                          label     index   +/-  runs")
        rows = sorted(zip(entries, runs, strict=True), key=lambda row: -row[0].total)
        for entry, count in rows:
            mark = "" if entry.viable else "  (gated out)"
            if entry.label == "human" and not entry.measurable:
                mark += f"  (ladder {entry.ladder:.2f}: unreadable by this roster)"
            # The pill's words (`web/static/page/core.js`'s `runsShort`), and no number at all
            # where the directory does not say how many runs it is the score of.
            shown = (
                f"{entry.total:6.3f} {entry.sd:5.3f}  {count} run{'' if count == 1 else 's'}"
                if count
                else f"{'no score':>12}"
            )
            print(f"  {entry.name[:32]:<33}{entry.label:<9}{shown}{mark}")
    return 0


def _runs_behind(run: Path, result: dict[str, Any]) -> int:
    """How many runs the index `_load_scored` read in ``run`` rests on, or 0 if unsaid.

    Read off what the directory holds. One with its games in it was scored from them,
    which is one tournament and so one run: every entry the labelled corpus writes, and
    every `runs/<hash>-<seed>/`. A measurement directory holds no games and names its
    seeds in `canonical.json`, which is believed only under the weights of the index file
    beside it. Anything else says nothing, and its row prints no number.
    """
    if (run / "games.jsonl").exists():
        return 1
    try:
        stored = json.loads((run / "canonical.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(stored, dict):
        return 0
    seeds = stored.get("seeds")
    if stored.get("weights_version") != result.get("weights_version"):
        return 0
    return len(seeds) if isinstance(seeds, list) else 0


def _load_scored(run: Path, weights: str | None) -> tuple[dict, str, str] | None:
    """Read one scored run and work out what label it carries.

    A mutant carries its own; otherwise a game is `control` or `human` by its name. The
    strength-premise check is not read: the hold that kept a failing game out of the band
    was withdrawn on 2026-10-01 (Q645, `docs/plan-confidence.md` §2.4).
    """
    candidates = sorted(run.glob("index-*.json"))
    if weights:
        candidates = [p for p in candidates if p.name == f"index-{weights}.json"]
    if not candidates:
        return None
    with open(candidates[-1], encoding="utf-8") as handle:
        result = json.load(handle)

    meta = run / "mutant.json"
    if meta.exists():
        with open(meta, encoding="utf-8") as handle:
            info = json.load(handle)
        return result, str(info["expectation"]), str(info["base"])

    name = str(result["ruleset_name"]).lower()
    if name in CONTROL_RULESETS:
        return result, "control", ""
    return result, "human", ""


def _cmd_anchor(args: argparse.Namespace) -> int:
    """Report where the band sits against a weights version's pinned curve ends.

    Since `v12` the band checks the scale and places none of it
    (`docs/plan-evolving-score.md` §4.2), so this writes nothing: it says, per component,
    which band games plateau, which score zero, and which ends a band game now passes. That
    is a check for the next revision, never a placement; a new end is a new weights version
    with its own pre-registration. Reads a directory of scored runs as `calibrate` does, or
    the library's `runs/`, where a `<hash>-<seed>` run beside its `<hash>` measurement is
    one of that measurement's runs and is not read twice.
    """
    from chess_variant_sandbox.metrics.calibration import Measured, check_ends
    from chess_variant_sandbox.metrics.scoring import load_weights

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    weights = load_weights(args.weights)

    corpus: list[Measured] = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        digest, _, seed = run.name.rpartition("-")
        if seed.isdigit() and (root / digest / "canonical.json").exists():
            continue
        # The raws do not depend on the weights, so a run scored under another version
        # still says where its game sits.
        scored = _load_scored(run, args.weights) or _load_scored(run, None)
        if scored is None:
            continue
        result, label, _ = scored
        corpus.append(Measured(name=result["ruleset_name"], label=label, raw=result["raw"]))

    checks = check_ends(corpus, weights)
    if args.json:
        print(json.dumps([check.as_dict() for check in checks], indent=2))
        return 0
    band = sum(1 for entry in corpus if entry.label == "human")
    print(f"the band ({band} games under {root}) against weights {weights.version}'s ends; "
          "nothing is written")
    for check in checks:
        ends = ", ".join(
            f"{end} {value:g} ({check.sources.get(end, 'band')})"
            for end, value in check.ends.items()
        )
        quantiles = (
            f"band {check.low:g} to {check.high:g}" if check.low is not None else "no band"
        )
        print(f"\n  {check.component} [{check.kind}] {ends}")
        print(f"    {check.games} readable, {check.plateau} on the plateau, {check.zero} at zero;"
              f" {quantiles} (k-th lowest and highest, k = ceil(0.1 n))")
        for end, names in check.past.items():
            if names:
                print(f"    past {end}: {', '.join(names)}")
        if check.unreadable:
            print(f"    unreadable ladder: {', '.join(check.unreadable)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
