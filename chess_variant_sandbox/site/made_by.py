"""Who designed a game, a person or one of the project's AI agents, as the site says it (L20).

`docs/plan-launch.md` R17: the site says who makes it, and which games an agent designed. A
game's page says it under the game's name (`detail.js` `designerByHtml`) and in *Where it came
from*, from `made_by`'s answer, which the site's `/api/game` sends as `madeBy`.

**Read off what is recorded, never guessed.** The facts come from the store and the game's
file, and from nothing else:

- a game somebody saved in the designer (`source` *designed*): the store's owner, a member of
  the site, named by their handle (`designer.by`). A person.
- `provenance.designed_by` (PV1), where the file records it, says it outright, one of
  `schema.DESIGNED_BY`: *people* (a person, named by `author` where the file names one),
  *agent*, *agent_variant* (an agent's variant of a game by people), *agent_from_owner* (an
  agent's build of the owner's idea; an agent, the owner credited) or *owner* (a person, the
  site's owner). It is read before `origin`, which it exists to settle; the evidence for each
  file is `docs/reference/game-provenance.md`.
- `provenance.origin` *generated* or *evolved*: a design run, which is an AI model. An agent.
- `provenance.origin` *designed*, *hand* or *control*: written for this project, and every hand
  that writes for this project is one of its AI agents, unless `provenance.author` names the
  person (*the user* is the site's owner). An agent, or that person.
- `provenance.origin` *human* with an `author` or a `year`: a published game, by the people it
  names or from before the project began. A person.
- anything else is **not recorded**, and the page says so. That covers *human* with neither,
  because the corpus has used *human* both for published games (chess, shogi) and for this
  project's own one-rule showcases written by its agents (`docs/archive/plan-roster-confound.md`
  §1 notes the same overload): the file cannot tell the two apart, so the page claims neither.
  Since PV1 every library file that origin left unsettled records `designed_by`, so this is
  left for a file that records neither.

Display only: nothing here reaches a score or a move.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

AGENT = "agent"
PERSON = "person"
UNRECORDED = "unrecorded"

OWNER_AUTHOR = "the user"
"""How the files name the site's owner as an author; said on the site as *the site's owner*,
so the page names nobody the site does not already name."""

AGENT_SAID = "designed by one of this project's AI agents"
DESIGN_RUN_SAID = AGENT_SAID + ", in a design run"
MEMBER_SAID = "designed by a member of this site, {by}"
MEMBER_UNNAMED_SAID = "built in the designer here"
OWNER_SAID = "designed by a person, the site's owner"
AUTHOR_SAID = "designed by a person, {author}"
AUTHORS_SAID = "designed by people: {author}"
BEFORE_SAID = "designed by people, before this project began"
PEOPLE_SAID = "designed by people, not by this project's AI agents"
VARIANT_SAID = (
    "designed by one of this project's AI agents, as a variant of a game designed by people")
FROM_OWNER_SAID = "designed by one of this project's AI agents, from the site's owner's idea"
UNRECORDED_SAID = (
    "not recorded: the records do not say whether a person or one of this project's AI "
    "agents designed it")

_RUNS = ("generated", "evolved")
_PROJECT = ("designed", "hand", "control")


def _said(text: Any) -> str:
    return " ".join(str(text).split()) if isinstance(text, str) else ""


def _by_author(author: str) -> dict[str, str]:
    if author.lower() == OWNER_AUTHOR:
        return {"kind": PERSON, "said": OWNER_SAID}
    many = " and " in author or "," in author
    return {"kind": PERSON, "said": (AUTHORS_SAID if many else AUTHOR_SAID).format(author=author)}


def _recorded(designed_by: Any, author: str) -> dict[str, str] | None:
    """What `provenance.designed_by` says, or `None` where the file records none (or a word
    the validator would refuse, which is then read as if absent)."""
    if designed_by == "people":
        if author and author.lower() != OWNER_AUTHOR:
            return _by_author(author)
        return {"kind": PERSON, "said": PEOPLE_SAID}
    if designed_by == "owner":
        return {"kind": PERSON, "said": OWNER_SAID}
    if designed_by == "agent":
        return {"kind": AGENT, "said": AGENT_SAID}
    if designed_by == "agent_variant":
        return {"kind": AGENT, "said": VARIANT_SAID}
    if designed_by == "agent_from_owner":
        return {"kind": AGENT, "said": FROM_OWNER_SAID}
    return None


def made_by(game: Mapping[str, Any]) -> dict[str, str]:
    """`{"kind": "agent" | "person" | "unrecorded", "said": <words>}` for a game's page payload
    (`library.detail`'s, before the site strips the maintainer's records: a design run's
    `author` is read here only to be ignored)."""
    if game.get("source") == "designed":
        designer = game.get("designer")
        by = _said(designer.get("by")) if isinstance(designer, Mapping) else ""
        if by:
            return {"kind": PERSON, "said": MEMBER_SAID.format(by=by)}
        return {"kind": UNRECORDED, "said": MEMBER_UNNAMED_SAID}
    found = game.get("provenance")
    provenance: Mapping[str, Any] = found if isinstance(found, Mapping) else {}
    origin = provenance.get("origin")
    author = _said(provenance.get("author"))
    recorded = _recorded(provenance.get("designed_by"), author)
    if recorded is not None:
        return recorded
    if origin in _RUNS:
        return {"kind": AGENT, "said": DESIGN_RUN_SAID}
    if origin in _PROJECT:
        return _by_author(author) if author else {"kind": AGENT, "said": AGENT_SAID}
    if origin == "human":
        if author:
            return _by_author(author)
        # A year is a number (1956) or words (*15th century*); both say it.
        if _said(str(provenance.get("year") or "")):
            return {"kind": PERSON, "said": BEFORE_SAID}
    return {"kind": UNRECORDED, "said": UNRECORDED_SAID}
