"""The site's terms, and the one sentence a designer agrees to when they publish (Q722).

The user, 2026-10-04: *"I have no intention of claiming any credit for them, I just want
people to be able to share and enjoy them. If people want to create something, withdraw it
as their own and sell it, sure fine (but can get a bit thorny if it's derived from
something else?). Just put something safely legalese distancing myself from
responsibility."*

So a published game goes to the public domain under CC0 1.0, the site keeps nothing and
claims no credit, the site is provided *as is*, and the designer answers for what they put
in. Each point is said twice on `/terms`: in plain words, and in the wording a lawyer would
read, side by side. **None of it is legal advice**; `docs/reference/terms-review.md` lists the
lines a lawyer should read before the site relies on them.

**The version is the record.** `VERSION` is what `drafts.publish` stamps on the game
(`games.terms_version`, `migrations/0029_terms.sql`) beside when it was accepted. A change
to anything a designer agrees to -- `DEDICATION` or a point's legal wording -- is a new
`VERSION` and a new `DATED`, never an edit under the old number, so a stored version always
names the words that were shown. A plain-words rewording that changes no meaning keeps the
number. Games published before the terms existed carry no version and are not marked
afterwards: nobody agreed to anything for them.

The contact for taking something down is `CONTACT`, the landing page's contact line
(Q720): the user's GitHub profile, never an email address. Naming it changed the plain
words only: the legal wording already sent notices "through the contact the operator names
on this page", and still does, so `VERSION` stays 1. Were `CONTACT` ever empty again, the
page would say how to ask without one, and invent no address.
"""

from __future__ import annotations

from dataclasses import dataclass

from chess_variant_sandbox.web import landing

VERSION = 1
"""The terms' version, stored on every game published under them."""

DATED = "4 October 2026"
"""When `VERSION` was written, as the page says it."""

PATH = "/terms"
"""Where the terms live. A stable address: the publish step and the privacy notice link it."""

CC0 = "https://creativecommons.org/publicdomain/zero/1.0/legalcode"
"""CC0 1.0's legal code, which the dedication names and the page links."""

CONTACT = landing.CONTACT
"""How to reach the person who runs the site: the landing page's contact line (Q720)."""

CONTACT_SAID = landing.CONTACT_SAID
"""`CONTACT` as the page writes it in a sentence."""

DEDICATION = (
    "By publishing, I dedicate this game's rules, name and description to the public domain "
    "under CC0 1.0, so anyone may copy, change, share or sell it, and I confirm I have the "
    "right to share what I put in."
)
"""The one sentence the publish step shows beside its box. Agreeing to it is agreeing to
`VERSION` of the terms."""

NOT_ACCEPTED = (
    "Tick the box to agree to the terms first: publishing gives the game to the public "
    "domain, and it cannot be published without that."
)
"""What a publish without the box ticked is told."""

STALE = (
    "The terms changed since this page was opened. Reload it, read the new sentence, and "
    "publish again."
)
"""What a publish agreeing to an older version is told. Nothing is published under words
the designer was not shown."""


@dataclass(frozen=True)
class Point:
    """One point of the terms: a heading, what it means in plain words, and its legal form."""

    key: str
    heading: str
    plain: str
    legal: str


def _takedown_plain() -> str:
    if CONTACT:
        return (
            "If something here is yours and was put in without your permission, or "
            "breaks the law, get in touch with the person who runs the site on GitHub, at "
            f"{CONTACT_SAID}, and say which game and why. Members can also report a game "
            "from its page. The person who runs the site reads every request and takes "
            "down what should not be here."
        )
    return (
        "If something here is yours and was put in without your permission, or breaks the "
        "law, say which game and why. Members can report a game from its page; anyone "
        "else can ask the person who invited them, and a contact address for the site "
        "will be named here. The person who runs the site reads every request and takes "
        "down what should not be here."
    )


def _takedown_legal() -> str:
    # The same words with a contact named or not: the page names it (`CONTACT_SAID`, in
    # the plain words and under the points), so the wording agreed to under `VERSION` 1
    # holds.
    where = (
        "through the report link on the game's page, or through the contact the "
        "operator names on this page"
    )
    return (
        "Notices of claimed infringement or unlawful content may be sent " + where + ", "
        "identifying the material, the right claimed and the claimant's contact details. "
        "The operator may remove or disable access to any material, and suspend any "
        "account, at its sole discretion and without notice, and is under no obligation "
        "to monitor material before or after it is published."
    )


def points() -> tuple[Point, ...]:
    """The terms, in the order the page shows them."""
    return (
        Point(
            "dedication",
            "Games you publish belong to everybody",
            "When you publish a game, you give it to the public domain. Its rules, its "
            "name and its description are then free for anyone to copy, change, share or "
            "sell, without asking and without naming you. The site keeps nothing for itself and claims no credit for your game. "
            "A draft you have not published stays yours. The site's own piece drawings "
            "and typefaces are not part of this: they keep the licences the credits page "
            "names.",
            "On publishing a game, the member who publishes it (the \"Creator\") "
            "dedicates the game's rules, name and description, and any other material "
            "the Creator publishes with it (the \"Work\"), to the public domain under the Creative "
            "Commons CC0 1.0 Universal Public Domain Dedication (" + CC0 + "), "
            "waiving all copyright and related or neighbouring rights in the Work "
            "worldwide to the extent permitted by law. Where such a waiver is not "
            "effective under applicable law, the Creator grants every person the "
            "royalty-free, non-transferable, non-sublicensable, non-exclusive, "
            "irrevocable and unconditional licence set out in section 3 of CC0 1.0 "
            "(License Fallback), and to the extent the Creator cannot waive or license "
            "any moral rights, agrees not to assert them. No right, title, licence or "
            "credit in the Work is reserved to or claimed by the site or its operator. "
            "Material supplied by the site itself, including its piece images and "
            "typefaces, is excluded and remains under its own licences.",
        ),
        Point(
            "yours",
            "What you put in is your responsibility",
            "Only put in what you have the right to share. The rules of a game are "
            "generally not anybody's property, so a variant of a game is fine. But "
            "names, drawings, written text, trade marks and other people's work under a "
            "licence can belong to someone, and you answer for them, not the site. If "
            "your game is built on somebody else's licensed work, their licence still "
            "applies to their part, and your dedication cannot take it away.",
            "By publishing, the Creator represents and warrants that they have all "
            "rights necessary to make the dedication above, and that the Work does not "
            "infringe any copyright, trade mark, database right, moral right or other "
            "right of any person, nor any law. The Creator acknowledges that, while game "
            "rules and mechanics are generally not protected by copyright, names, "
            "artwork, text, trade marks and third-party licensed material may be, and "
            "that such material is the Creator's sole responsibility. Where the Work "
            "incorporates or is derived from third-party material made available under "
            "a licence, that material remains subject to its licence, which the "
            "dedication does not and cannot override; the Creator is responsible for "
            "complying with it. The Creator agrees to indemnify and hold harmless the "
            "operator against claims arising from material the Creator supplies, to the "
            "extent permitted by law.",
        ),
        Point(
            "as-is",
            "The site comes as it is",
            "The site, its games, its scores, its computer players and everything else "
            "on it are offered as they are, with no promise that they are right, safe, "
            "available or fit for anything. It is a small site, and it may change, "
            "break, lose things or close.",
            "The site and all material, games, scores, ratings, software and services "
            "on or through it are provided \"as is\" and \"as available\", without "
            "warranty of any kind, express or implied, including without limitation any "
            "warranty of merchantability, fitness for a particular purpose, accuracy, "
            "availability, non-infringement or title. The operator may change, suspend "
            "or discontinue the site or any part of it, or delete any material, at any "
            "time without notice or liability.",
        ),
        Point(
            "members",
            "People answer for themselves",
            "Members make games, play and post suggestions on their own account. The "
            "person who runs the site is not responsible for what members make, say or "
            "do, here or anywhere else, or for what anybody does with a game they found "
            "here.",
            "To the fullest extent permitted by law, the operator is not responsible or "
            "liable for any content, conduct or communication of any member or third "
            "party, on or off the site, nor for any use made of any material obtained "
            "through the site, and shall not be liable for any direct, indirect, "
            "incidental, special, consequential or exemplary loss or damage arising out "
            "of or in connection with the site or its use. Nothing in these terms "
            "excludes or limits liability which cannot be excluded or limited under "
            "applicable law.",
        ),
        Point(
            "takedown",
            "Asking for something to come down",
            _takedown_plain(),
            _takedown_legal(),
        ),
        Point(
            "changes",
            "When these terms change",
            "If the terms change, this page gets a new version and date. A game keeps "
            "the version it was published under, which the site records when you "
            "publish; a dedication, once made, cannot be taken back.",
            "The operator may amend these terms by publishing a new version on this "
            "page. Each publication of a game is governed by the version accepted at "
            "that time, which the operator records. A public domain dedication made "
            "under any version is irrevocable.",
        ),
    )
