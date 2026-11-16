"""R-W7: the designer sends a signed-out visitor's page no request the site refuses.

On the live site a visitor's designer asked `GET /api/design/standing/<hash>` and
`POST /api/design/screen` for every draft, and the site answered both 400: each needs a
signed-in person (`site/routes.py`, their `visitor` sentences). The page drew the refusals
as a failed rung and a save rung *asking...* for ever. Now the page asks neither for a
visitor and its rungs say what signing in opens (`design.js` `signedOutHere`); a signed-in
person and the local application ask both as before. Run in node on the page's own script.
"""

from __future__ import annotations

from typing import Any

from test_wood import _run

from chess_variant_sandbox.site import routes

_HASH = "831c77e75650"


def _asked(accounts: bool, viewer: str | None) -> dict[str, Any]:
    """The requests the designer sends for a new draft hash, and the rungs it then draws."""
    who = "null" if viewer is None else repr(viewer)
    return dict(_run(
        "(() => { const asked = [];"
        " global.fetch = (url, how) => { asked.push(((how && how.method) || 'GET') + ' '"
        "   + String(url).split('?')[0]); return new Promise(() => {}); };"
        f" ACCOUNTS = {'true' if accounts else 'false'}; VIEWER = {who};"
        f" DRAFT_HASH = '{_HASH}'; DRAFT = {{ name: 'x' }};"
        f" loadStanding('{_HASH}'); runScreen('{_HASH}');"
        f" const check = {{ valid: true, hash: '{_HASH}', playable: true, smokeSeconds: 0.1 }};"
        " const states = rungStates(check, null, DRAFT_SCREEN);"
        " return { asked, screen: states.screen, save: states.save }; })()"))


def _visitor_only(method: str, path: str) -> bool:
    route = routes.find(method, path)
    return route is not None and bool(route.visitor)


class TestAVisitorsDesigner:
    def test_the_two_requests_are_ones_the_site_refuses_a_visitor(self) -> None:
        """The premise: both routes carry a visitor sentence, which the site answers 400."""
        assert _visitor_only("GET", f"/api/design/standing/{_HASH}")
        assert _visitor_only("POST", "/api/design/screen")

    def test_a_visitor_on_the_site_is_sent_neither(self) -> None:
        said = _asked(accounts=True, viewer=None)
        assert said["asked"] == []

    def test_a_visitors_rungs_say_what_signing_in_opens_and_nothing_failed(self) -> None:
        said = _asked(accounts=True, viewer=None)
        assert said["screen"]["mark"] == "ahead" and "sign in" in said["screen"]["says"]
        assert said["save"] == {"mark": "ahead", "says": "sign in to save it"}

    def test_a_signed_in_person_is_sent_both(self) -> None:
        said = _asked(accounts=True, viewer="someone")
        assert f"GET /api/design/standing/{_HASH}" in said["asked"]
        assert "POST /api/design/screen" in said["asked"]
        assert said["screen"]["mark"] == "running" and said["save"]["mark"] == "running"

    def test_the_local_application_is_sent_both(self) -> None:
        said = _asked(accounts=False, viewer=None)
        assert f"GET /api/design/standing/{_HASH}" in said["asked"]
        assert "POST /api/design/screen" in said["asked"]
