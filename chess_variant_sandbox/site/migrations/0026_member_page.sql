-- A member's page and the vote's fixed tags (Q596 and Q599, `docs/plan-return.md` R1 and R4).
--
-- **`users.show_favourites`** is the box on your own page, *show my five-star games on my
-- page*, **off by default**: a vote is private until its voter ticks it, and then only
-- the games they gave five stars are named, to members, on `/players/<handle>`. Leaving
-- clears it with the address.
--
-- **The vote holds no text from its voter** (the user, 2026-10-02: *"Votes only, no
-- comments"*). `votes.note` stays a column, so no row is deleted, but no new or changed
-- row may hold one; `votes.tags` may hold only the eight fixed tags of `site/votes.py`
-- (`TAGS`), at most three. Both are `not valid`: they bind every write from now on and
-- read nothing already stored, and the readers count only the eight, so a stray old
-- string is never shown. A sentence a voter writes cannot be stored, which is what makes
-- *no notes to a designer* a fact about the table rather than about a page.

alter table users add column show_favourites boolean not null default false;

alter table votes add constraint votes_no_note check (note is null) not valid;

alter table votes add constraint votes_fixed_tags check (
    cardinality(tags) <= 3
    and tags <@ array[
        'again', 'clever', 'first-easy', 'draws', 'long', 'fast', 'hard-to-learn', 'new'
    ]::text[]
) not valid;
