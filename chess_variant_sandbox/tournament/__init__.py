"""Running games at scale and recording what happened.

Simulation writes records; analysis reads them. Metrics never touch a live engine, so
the index can be recomputed under new weights without replaying a tournament.

Games run inside the Rust core, on threads. Every game's seed is fixed by the schedule
before any thread starts, so the corpus is identical at any worker count -- that is what
makes ``--workers`` a speed knob and not an experimental variable.
"""

from chess_variant_sandbox.tournament.play import (
    choose_move,
    finish_game,
    play_game,
    play_game_mixed,
    play_game_python,
    replay,
    seat_rng,
)
from chess_variant_sandbox.tournament.ratings import RatingTable, Standing, expected_score, rate
from chess_variant_sandbox.tournament.record import GameRecord, read_records, write_records
from chess_variant_sandbox.tournament.runner import (
    TournamentConfig,
    TournamentResult,
    run_tournament,
    schedule,
)

__all__ = [
    "GameRecord",
    "RatingTable",
    "Standing",
    "TournamentConfig",
    "TournamentResult",
    "choose_move",
    "expected_score",
    "finish_game",
    "play_game",
    "play_game_mixed",
    "play_game_python",
    "rate",
    "read_records",
    "replay",
    "run_tournament",
    "schedule",
    "seat_rng",
    "write_records",
]
