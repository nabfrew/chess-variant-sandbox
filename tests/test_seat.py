"""The seat gate's pooled reading and the transcript digest, pinned to literal values.

Every band here is built inline; no run directory and no engine is read.
"""

from __future__ import annotations

from chess_variant_sandbox.metrics.seat import (
    SEAT_GATE,
    TOP_RUNG_SEAT_DETAIL,
    TOP_THREE_SEAT_DETAIL,
    pooled_seat_reading,
    transcript_digest,
)


def test_transcript_digest_is_sixteen_stable_hex_digits() -> None:
    # Load-bearing: runs store these digests and the pooled reading unions them across
    # runs scored at different times, so a changed digest would split one game in two.
    assert transcript_digest([1, 2, 3]) == "e2353910701f1c2a"


def test_pooled_reading_of_a_tied_band_with_keys_missing_from_some_runs() -> None:
    # Run one stores no draws or games, run two no wins or losses: each absent count is 0.
    reading = pooled_seat_reading(
        [
            {TOP_RUNG_SEAT_DETAIL: {"wins": 3, "losses": 3, "rungs": [1600, 400]}},
            {TOP_RUNG_SEAT_DETAIL: {"draws": 2, "games": 8, "rungs": [1600]}},
        ]
    )
    assert reading == {
        "tolerated_share": 0.6667,
        "bands": {
            TOP_RUNG_SEAT_DETAIL: {
                "component": "seat_decided_chance",
                "rungs": [400, 1600],
                "runs": 2,
                "games": 8,
                "wins": 3,
                "losses": 3,
                "draws": 2,
                "decided": 6,
                "distinct": None,
                "favoured": "first",
                "share": 0.5,
                "counted": "as played",
                "distinct_wins": None,
                "distinct_losses": None,
                "tested_favoured": "first",
                "tested_share": 0.5,
                "chance": 1.0,
                "alpha": 0.001,
                "refuses": False,
                "refusal": "",
                "lean": "",
            }
        },
        "refuses": False,
        "violations": [],
        "lean": "",
    }


def test_pooled_reading_of_a_band_with_no_decided_game() -> None:
    reading = pooled_seat_reading(
        [{TOP_RUNG_SEAT_DETAIL: {"wins": 0, "losses": 0, "draws": 4, "games": 4, "rungs": [100]}}]
    )
    band = reading["bands"][TOP_RUNG_SEAT_DETAIL]
    assert (band["decided"], band["favoured"], band["share"]) == (0, "", 0.5)
    assert (band["tested_favoured"], band["tested_share"], band["chance"]) == ("", 0.5, 1.0)


def test_pooled_reading_of_a_lone_wide_band_tested_on_distinct_games() -> None:
    # No run measured the pair: the wide band is still read. Played 4-2, but the three
    # different games split 1-2, so the test and its share read the second seat.
    reading = pooled_seat_reading(
        [
            {
                TOP_THREE_SEAT_DETAIL: {
                    "wins": 4,
                    "losses": 2,
                    "draws": 0,
                    "games": 6,
                    "rungs": [100],
                    "transcripts": ["a", "b", "c"],
                    "first_seat_transcripts": ["a"],
                }
            }
        ]
    )
    assert list(reading["bands"]) == [TOP_THREE_SEAT_DETAIL]
    band = reading["bands"][TOP_THREE_SEAT_DETAIL]
    assert {
        key: band[key]
        for key in (
            "favoured",
            "share",
            "counted",
            "distinct",
            "distinct_wins",
            "distinct_losses",
            "tested_favoured",
            "tested_share",
            "alpha",
        )
    } == {
        "favoured": "first",
        "share": 0.6667,
        "counted": "distinct",
        "distinct": 3,
        "distinct_wins": 1,
        "distinct_losses": 2,
        "tested_favoured": "second",
        "tested_share": 0.6667,
        "alpha": 0.0005,
    }


def test_pooled_refusal_quotes_the_distinct_decided_count() -> None:
    digests = [str(i) for i in range(32)]
    reading = pooled_seat_reading(
        [
            {
                TOP_RUNG_SEAT_DETAIL: {
                    "wins": 35,
                    "losses": 2,
                    "draws": 1,
                    "games": 38,
                    "rungs": [100, 200],
                    "transcripts": digests,
                    "first_seat_transcripts": digests[:30],
                }
            }
        ]
    )
    assert reading["refuses"] is True
    assert reading["violations"] == [
        "seat_decided_chance=3.27e-04 below minimum 0.001: the first seat took 94% of "
        "the 32 decided games, each different game counted once (37 as played), among "
        "the 38 played between the two strongest rungs (100, 200 nodes) over 1 runs, "
        f"which rules out a seat held to 67% or less ({SEAT_GATE.reason})"
    ]
