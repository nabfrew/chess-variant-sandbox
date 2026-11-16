"""R-W7: the opponent picker's cards have names a screen reader can say.

Each seat's row of cards was a word and a line under it (*hard*; *greedy*, *gentlest*), the
same words on both seats, and the cards carried no name of their own, so the live site's
sixteen buttons read as sixteen nameless or doubled buttons. Each card now names itself
whole (`detail.js` `seatCardName`): whose seat, which player, and how strong in the server's
own plain words. Checked here on the markup the page draws from the real ladder; the
accessibility tree was read in a browser from a served worktree (R-W7's commit).
"""

from __future__ import annotations

import json
import re

from test_wood import _run

from chess_variant_sandbox.web import opponents

_GAME = {"hash": "abc", "sides": ["white", "black"], "armies": [[], []]}


def _drawn(ladder: list[dict[str, object]], game: dict[str, object] = _GAME) -> str:
    return str(_run(f"(() => {{ ROSTER = {json.dumps(ladder)};"
                    f" return sidePicker({json.dumps(game)}, ROSTER, null); }})()"))


def _names(drawn: str) -> list[str]:
    cards = re.findall(r'<button type="button" class="seatCard"[^>]*>', drawn)
    found = [re.search(r'aria-label="([^"]*)"', card) for card in cards]
    assert all(found), cards
    return [m.group(1) for m in found if m]


def _ladder() -> list[dict[str, object]]:
    return [o.as_dict() for o in opponents.ladder()]


class TestTheSeatCardsNames:
    def test_sixteen_cards_sixteen_different_names(self) -> None:
        names = _names(_drawn(_ladder()))
        assert len(names) == 2 * len(opponents.RUNGS) == 16
        assert len(set(names)) == len(names)

    def test_each_name_says_the_seat_the_player_and_how_strong(self) -> None:
        ladder = _ladder()
        names = _names(_drawn(ladder))
        for seat, side in enumerate(("white", "black")):
            for at, option in enumerate(ladder):
                name = names[seat * len(ladder) + at]
                assert name.startswith(f"{side}: "), name
                if option["agent"] == opponents.HUMAN:
                    assert "you" in name
                    continue
                assert f"{side}: {option['name']}, computer player" in name
                # How strong, in the server's words (`web.play.strength`): never empty on a rung.
                assert option["about"] and str(option["about"]) in name, name

    def test_the_gentlest_and_a_games_own_side_names_are_said(self) -> None:
        game = dict(_GAME, sides=["first", "second"],
                    firstGame={"gentle": {"agent": "greedy", "name": "greedy"}})
        names = _names(_drawn(_ladder(), game))
        assert all(n.startswith(("first: ", "second: ")) for n in names)
        gentle = [n for n in names if "the gentlest on this game" in n]
        assert gentle and all(", greedy," not in n and n.split(": ")[1].startswith("greedy")
                              for n in gentle)

    def test_each_seats_row_is_a_named_group(self) -> None:
        drawn = _drawn(_ladder())
        assert 'aria-label="who plays white"' in drawn
        assert 'aria-label="who plays black"' in drawn
