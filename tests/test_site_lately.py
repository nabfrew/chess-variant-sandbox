"""*Lately* on your own page (UP3, `docs/plan-user-page.md` §5, choice C1 of §4).

One reader over your games between people, challenges, milestones and badges, and votes
and first plays on the games you made (`site/lately.py`): dated, newest first, the last
`lately.LIMIT` at most, each a sentence in the site's words. Nothing unread, nothing
counted, no last visit read or sent; no line about another person's private game.

The store tests need `CHESS_VARIANT_SANDBOX_TEST_DATABASE_URL` and skip without it, as
every site test does. Proved on specs rather than chess (`test_site_matches.py`'s games).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from test_site import Site, _joined, _rendered, _running_site
from test_site_matches import LADDER, SHORT, SHORT_HASH, UNRATED, UNRATED_HASH, Scratch, _scratch
from test_site_members import _designed, _play_out

from chess_variant_sandbox.site import awards, badges, challenges, lately, milestones
from chess_variant_sandbox.site import correspondence as c
from chess_variant_sandbox.site import pages as site_pages
from chess_variant_sandbox.site import server as site_server

DAY = datetime(2026, 10, 1, tzinfo=timezone.utc)
NAME = SHORT["name"]


@pytest.fixture
def store() -> Iterator[Scratch]:
    with _scratch() as scratch:
        scratch.game(SHORT_HASH, SHORT, LADDER)
        scratch.game(UNRATED_HASH, UNRATED)
        yield scratch


def _said(line: dict[str, Any]) -> str:
    """A line as the page reads it."""
    return "".join(
        part if isinstance(part, str) else part.get("name") or part.get("person", "")
        for part in line["parts"]
    )


def _challenge(store: Scratch, sender: int, to: int, game: str, made: datetime,
               answer: str | None = None) -> None:
    with store.connection.cursor() as cursor:
        cursor.execute(
            "insert into challenges (challenger, challenged, game, made, answer, answered)"
            " values (%s, %s, %s, %s, %s, %s)",
            (sender, to, game, made, answer, None if answer is None else made),
        )
    store.connection.commit()


def _match(store: Scratch, game: str, first: int, second: int | None, started: datetime) -> None:
    """A practice game on `game`, unfinished: all a first play reads."""
    with store.connection.cursor() as cursor:
        cursor.execute(
            "insert into matches (id, game, mode, seed, seat0_user, seat1_user, seat1_agent,"
            " started) values (%s, %s, 'sandbox', 0, %s, %s, %s, %s)",
            (f"m{first}{game}{started.timestamp():.0f}", game, first, second,
             None if second is not None else "random", started),
        )
    store.connection.commit()


def _vote(store: Scratch, voter: int, game: str, when: datetime) -> None:
    with store.connection.cursor() as cursor:
        cursor.execute(
            "insert into votes (voter, game, stars, updated) values (%s, %s, 4, %s)",
            (voter, game, when),
        )
    store.connection.commit()


class TestTheReader:
    def test_newest_first_and_never_more_than_the_bound(self, store: Scratch) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        for day in range(lately.LIMIT + 5):
            _challenge(store, bea, ada, SHORT_HASH, DAY + timedelta(days=day))
        found = lately.read(store.connection, ada)
        assert len(found) == lately.LIMIT
        whens = [line["when"] for line in found]
        assert whens == sorted(whens, reverse=True)
        assert found[0]["when"] == (DAY + timedelta(days=lately.LIMIT + 4)).isoformat()
        assert _said(found[0]) == f"bea challenged you to {NAME}."
        assert {key for line in found for key in line} == {"when", "kind", "parts"}

    def test_each_store_says_its_line_and_they_interleave_by_date(self, store: Scratch) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        mine = _designed(store, ada, "one stone", public=True)
        _challenge(store, bea, ada, SHORT_HASH, DAY)
        _challenge(store, ada, bea, SHORT_HASH, DAY + timedelta(days=1), challenges.DECLINED)
        _match(store, mine, bea, None, DAY + timedelta(days=2))
        _match(store, mine, bea, None, DAY + timedelta(days=2, hours=1))
        _vote(store, bea, mine, DAY + timedelta(days=3))
        _vote(store, ada, mine, DAY + timedelta(days=3, hours=1))
        held = [awards.Award(kind=badges.BEAT[0], game=SHORT_HASH, earned=DAY + timedelta(days=4))]
        lines = [milestones.Milestone(kind=milestones.FIRST_PERSON, when=DAY + timedelta(days=5),
                                      game=SHORT_HASH, seat=0, match="x"),
                 # D3 is the first of the first plays, which say it; never twice.
                 milestones.Milestone(kind=milestones.PLAYED_BY_OTHERS,
                                      when=DAY + timedelta(days=2), game=mine, seat=-1,
                                      match="y")]
        _play_out(store, c.start(store.connection, SHORT_HASH, [ada, bea]).id)
        found = lately.read(store.connection, ada, lines=lines, held=held)
        said = [_said(line) for line in found]
        assert said[0] in {f"You beat bea at {NAME}.", f"bea beat you at {NAME}.",
                           f"You drew with bea at {NAME}."}
        assert said[1:3] == [f"Against a person, on {NAME}.",
                             f"Badge: {badges.name_for(held[0])}, on {NAME}."]
        # A vote names nobody, and your own vote on your own game is no news.
        assert said[3:] == [
            "Somebody voted on one stone.",
            "Somebody played one stone for the first time.",
            f"bea declined your challenge to {NAME}.",
            f"bea challenged you to {NAME}.",
        ]
        kinds = [line["kind"] for line in found]
        assert kinds[0] == lately.FINISHED and kinds[-1] == lately.CHALLENGED
        assert set(kinds) <= set(lately.KINDS)

    def test_a_running_game_says_whose_move_it_is_from_the_profiles_entry(
        self, store: Scratch
    ) -> None:
        ada, bea = store.user("ada"), store.user("bea")
        found = c.start(store.connection, SHORT_HASH, [ada, bea])
        store.connection.commit()
        clock = c.now(store.connection)
        entries = [site_server._between_entry(store.connection, found, ada, clock)]
        assert [_said(line) for line in lately.read(store.connection, ada, running=entries)] \
            == [f"Your game of {NAME} against bea began."]
        moved = [{**entries[0], "plies": 1, "yourMove": False}]
        assert [_said(line) for line in lately.read(store.connection, ada, running=moved)] \
            == [f"A move in {NAME} against bea; it is their move."]

    def test_no_line_about_another_persons_private_game(self, store: Scratch) -> None:
        """Somebody's draft is named nowhere in your list -- not a game you played on its
        link, not a challenge to it -- and nobody who played or voted on yours is named."""
        ada, bea, cy = store.user("ada"), store.user("bea"), store.user("cy")
        theirs = _designed(store, bea, "bea's secret draft", public=False)
        mine = _designed(store, ada, "ada's draft", public=False)
        _challenge(store, bea, ada, theirs, DAY)
        _match(store, theirs, ada, bea, DAY + timedelta(days=1))
        _match(store, mine, bea, cy, DAY + timedelta(days=2))
        _vote(store, cy, mine, DAY + timedelta(days=3))
        found = lately.read(store.connection, ada)
        text = json.dumps(found)
        assert theirs not in text and "secret" not in text
        assert "bea" not in text and "cy" not in text
        # Your own draft is yours to hear about.
        assert [_said(line) for line in found] == [
            "Somebody voted on ada's draft.",
            "Somebody played ada's draft for the first time.",
            "Somebody played ada's draft for the first time.",
        ]


@pytest.fixture
def site() -> Iterator[Site]:
    with _running_site() as running:
        yield running


class TestThePayload:
    def test_lately_is_sent_and_nothing_names_a_last_visit(self, site: Site) -> None:
        browser = _joined(site, "ada")
        payload = browser.get("/api/profile").payload
        assert payload["lately"] == []

        def keys(value: Any) -> Iterator[str]:
            if isinstance(value, dict):
                for key, inner in value.items():
                    yield str(key)
                    yield from keys(inner)
            elif isinstance(value, list):
                for inner in value:
                    yield from keys(inner)

        found = [key for key in keys(payload)
                 if re.search(r"visit|seen|unread|since", key, re.IGNORECASE)]
        assert found == []

    def test_the_reader_reads_no_visit(self) -> None:
        source = lately.__file__
        with open(source, encoding="utf-8") as handle:
            code = "".join(line for line in handle if not line.lstrip().startswith("#"))
        body = code[code.index('"""', 3) + 3:]
        assert "sessions" not in body and "seen" not in body and "localStorage" not in body


LINES = [
    {"when": "2026-10-26T10:00:00+00:00", "kind": "moved",
     "parts": ["A move in ", {"game": "gggg", "name": "five stones"}, " against ",
               {"person": "lin"}, "; it is your move."]},
    {"when": "2026-10-24T09:00:00+00:00", "kind": "voted",
     "parts": ["Somebody voted on ", {"game": "gggg", "name": "five stones"}, "."]},
]


def _drawn(lines: list[dict[str, Any]]) -> Any:
    return _rendered(
        f"drawLately({json.dumps(lines)});"
        "const list = document.getElementById('late-lines');"
        "const links = [];"
        "(function walk(node) {"
        "  if (node.href !== undefined) links.push([node.href, node.textContent]);"
        "  (node.children || []).forEach(walk);"
        "})(list);"
        "console.log(JSON.stringify({rows: list.children.map(row => row.textContent), links,"
        " hidden: document.getElementById('lately').hidden}));"
    )


class TestThePage:
    def test_each_line_is_its_day_then_the_servers_sentence_with_its_links(self) -> None:
        drawn = _drawn(LINES)
        assert drawn["rows"] == [
            "26 Oct 2026A move in five stones against lin; it is your move.",
            "24 Oct 2026Somebody voted on five stones.",
        ]
        assert ["/players/lin", "lin"] in drawn["links"]
        assert any(link[1] == "five stones" for link in drawn["links"])
        assert drawn["hidden"] is False

    def test_the_newest_lead_and_the_rest_are_under_earlier_never_counted(self) -> None:
        lines = [{**LINES[1], "when": f"2026-10-{20 - day:02d}T09:00:00+00:00"}
                 for day in range(5)]
        drawn = _rendered(
            f"drawLately({json.dumps(lines)});"
            "const more = document.getElementById('late-more');"
            "console.log(JSON.stringify([document.getElementById('late-lines').children.length,"
            " document.getElementById('late-earlier').children.length, more.hidden,"
            " more.textContent]));"
        )
        assert drawn[:3] == [2, 3, False]
        assert not re.search(r"\d+ (more|earlier|lines)", drawn[3])
        assert _rendered(
            f"drawLately({json.dumps(LINES)});"
            "console.log(JSON.stringify(document.getElementById('late-more').hidden));"
        ) is True

    def test_with_no_line_there_is_no_heading(self) -> None:
        assert _drawn([])["hidden"] is True

    def test_it_stands_beside_the_record_and_marks_nothing(self) -> None:
        page = site_pages.profile("ada")
        at = [page.index(mark) for mark in
              ('id="mine"', 'id="lately"', 'id="record-part"', 'id="done"')]
        assert at == sorted(at)
        script = (site_pages.SCRIPT_DIR / "profile.js").read_text(encoding="utf-8")
        body = script[script.index("function drawLately"):]
        body = body[: body.index("\n}\n")]
        for word in ("localStorage", "unread", "length +", "count("):
            assert word not in body, word
