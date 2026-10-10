-- A visitor asks for an invitation (Q831, `docs/plan-website.md`, *Public or invite-only,
-- indefinitely*, decided 2026-10-10).
--
-- The user: *"Make a request invite button, with an optional message, suggesting they say
-- how they found the site."* Accounts stay by invitation. A visitor who is not signed in
-- leaves an address and, if they like, a note; the row is read on `/admin`, where the
-- maintainer grants it (the ordinary invitation, `identity.send_invitation`, with the
-- maintainer as its issuer) or declines it. Nothing is mailed when a request arrives, and
-- the visitor is told nothing about the outcome but the invitation, if one comes.
--
-- **One row an address** is the unique index, so it holds of the table rather than of the
-- one writer: a second request from the same address, while the first is kept, writes
-- nothing (`asking.ask` inserts `on conflict do nothing`), and the visitor reads the same
-- sentence either way. An answered row is deleted `quota.ASK_KEPT` after its answer, so an
-- address may ask again after that and nothing about it is kept longer than it is useful.
--
-- `note` is optional and short, the rule `reports.note` follows: the cap is a check
-- constraint so that it holds of the table. `email` is lowered by `identity`'s address
-- check before it is written, and capped at the length an address can have.
--
-- `answered` and `granted` are set together or not at all: a pending request has neither.
-- No column names a user, because the person asking has no account; the maintainer who
-- answered is the invitation's issuer, on `invites`, when the answer was yes.

create table invite_asks (
    id       bigint generated always as identity primary key,
    email    text not null check (email <> '' and length(email) <= 254),
    note     text check (note is null or (note <> '' and length(note) <= 500)),
    created  timestamptz not null default now(),
    answered timestamptz,
    granted  boolean,
    check ((answered is null) = (granted is null))
);

create unique index invite_asks_one_an_address on invite_asks (email);

-- The rolling day `quota.ASKS_A_DAY` counts over, and the sweep of answered rows.
create index invite_asks_by_created on invite_asks (created);
