-- The suggestion box: an idea for a rule, or an idea for the site.
--
-- The user's inbox, 2026-09-17: *"Add a place to add ideas of game-rules or other website
-- ideas that could be added."* So a signed-in person writes a short text, picks one of two
-- kinds, and it lands here. Nothing reads this table automatically; the box is for people.
--
-- `author` does not cascade, which is `awards.holder`'s rule and `matches.seat0_user`'s.
-- Leaving empties the person out of their rows rather than removing them
-- (`0005_leaving.sql`), so a suggestion by somebody who has gone reads as *a former
-- player* through the same `coalesce` every other reader uses. A suggestion is about the
-- site rather than about its author, and removing it would take away an idea the
-- maintainer may already have acted on.
--
-- `kind` is closed to two values because the page offers two radio buttons and a third
-- string could only come from something other than the page. Open text is `body`, capped
-- in the column as well as in `site/quota.py`: the constraint is what makes the cap true
-- of the table rather than of one writer, the same way `games_deleted_has_no_spec` makes
-- *the rules are gone* structural.
--
-- `done` is an admin's mark and is reversible; there is no third state, because a
-- suggestion that is neither open nor done is one somebody deleted, and a delete here is
-- a real `delete` -- nothing references a suggestion, so no count moves when one goes.

create table suggestions (
    id      bigint generated always as identity primary key,
    author  bigint not null references users (id),
    kind    text not null check (kind in ('rule', 'site')),
    body    text not null check (body <> '' and length(body) <= 2000),
    created timestamptz not null default now(),
    done    timestamptz
);

-- The two reads: this person's own suggestions on the suggestions page, and the rolling
-- day `quota.suggestions_today` counts over. Both are by author, so one index serves them.

create index suggestions_by_author on suggestions (author, created);
