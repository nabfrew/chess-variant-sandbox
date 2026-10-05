"""No page script uses a name that nothing declares.

`docs/reference/page-javascript.md` §3 and §5: of the seventeen page defects git history
holds, two are an undefined name, and both passed `node --check` and every test of their
day. Each is reconstructed here from today's code by taking away the one declaration it
lacked, so the check is held to catching the slip that actually happened, and today's
scripts are held to having none. The walk and its parser are `tests/page_names.py`'s.
"""

from __future__ import annotations

import node_session
import page_names


def _names(found: object) -> set[str]:
    assert isinstance(found, list), found
    return {one["name"] for one in found}


def _without(source: str, declaration: str) -> str:
    """`source` with the one declaration that starts `declaration` renamed away."""
    assert source.count(declaration) == 1, declaration
    return source.replace(declaration, declaration.replace("(", "Gone("))


class TestNoPageScriptUsesAnUndeclaredName:
    def test_every_page_script_is_clean(self) -> None:
        sources = page_names.scripts()
        found = page_names.undefined(sources)
        assert set(found) == set(sources)
        dirty = {label: names for label, names in found.items() if names}
        assert not dirty, dirty

    def test_every_page_script_is_walked(self) -> None:
        """The list is found, not written; this holds it to what is known to be there."""
        assert set(page_names.scripts()) >= {
            "the page", "the replay", "/join", "/signin", "/profile", "/players",
            "/suggestions", "/admin",
        }

    def test_the_router_calling_views_not_yet_written_is_caught(self) -> None:
        """`bf0fee37` (08-30): the router sent `#/design` and `#/opponent` to
        `showDesigner` and `showBotBuilder` a commit before either existed."""
        page = page_names.scripts()["the page"]
        page = _without(page, "async function showDesigner(")
        page = _without(page, "async function showBotBuilder(")
        assert node_session.run(page, check=True).returncode == 0
        found = page_names.undefined({"the page": page})["the page"]
        assert _names(found) == {"showDesigner", "showBotBuilder"}

    def test_a_helper_only_another_pages_script_defines_is_caught(self) -> None:
        """`12d8df4a` (09-19): the admin page called `span()`, which only the profile's
        script defined, so a deleted game's row stopped the table drawing."""
        sources = page_names.scripts()
        assert "function span(" in sources["/profile"]
        admin = _without(sources["/admin"], "function span(")
        assert node_session.run(admin, check=True).returncode == 0
        found = page_names.undefined({"/admin": admin})["/admin"]
        assert _names(found) == {"span"}

    def test_the_walk_follows_the_languages_scoping(self) -> None:
        """What the walk declares and what it reads, one construct each: a false
        positive would fail today's pages, a false negative would pass a slip."""
        declared = """
            function hoisted() { return later + inBlock + arguments.length; }
            var later = 1;
            if (later) { var inBlock = 2; function annexB() {} }
            annexB();
            const { a, b: [c, ...d] = [], ...e } = {};
            for (const f of [a, c, d, e]) f;
            for (let g = 0; g < 1; g++) g;
            try { hoisted(); } catch ({ message }) { message; }
            outer: for (;;) { break outer; }
            class K { #p = 1; static s = K; m() { return this.#p + K.s; } }
            const h = function named() { return named; };
            ({ h, key: typeof nowhere });
            switch (later) { case 1: let inCase = 1; inCase; }
            new K(); h.anything; window.document; JSON; Promise; cytoscape;
        """
        assert page_names.undefined({"declared": declared}) == {"declared": []}
        read = """
            const { x = missingDefault, [missingKey]: y } = {};
            for (const z of missingIterable) z;
            missingTarget = 1;
            ({ missingShorthand, key: missingValue });
            class L { f = missingField; }
            { let blockOnly; }
            blockOnly;
            function inner() { let own; } own;
        """
        found = page_names.undefined({"read": read})["read"]
        assert _names(found) == {
            "missingDefault", "missingKey", "missingIterable", "missingTarget",
            "missingShorthand", "missingValue", "missingField", "blockOnly", "own",
        }

    def test_a_script_that_does_not_parse_says_so(self) -> None:
        assert "error" in page_names.undefined({"broken": "let ("})["broken"]
