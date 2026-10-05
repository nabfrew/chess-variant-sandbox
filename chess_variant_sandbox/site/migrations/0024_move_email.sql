-- Correspondence play, *your move* (Q572, `docs/plan-return.md` §1.5).
--
-- The one email the site sends besides an invitation: when a move lands that makes it a
-- person's move in a game between two people, to a person who switched it on. The dated
-- exception to test 3 of `docs/plan-meta-game.md` (the user, 2026-10-01).
--
-- **`users.move_email`** is the box on a person's own page, **off by default**.
-- **`users.mail_off`** is the token in every move email's one-click *switch them off*
-- link (`/mail/off/<token>`). It is made the first time the box is ticked and kept, so a
-- link in an old message still works after the box is ticked again; it is no sign-in and
-- can do nothing but switch the box off. Leaving clears both with the address.
--
-- **`matches.seen0` and `seen1`** are when each seat's player last read the match while it
-- ran. The match page asks for the match every 15 seconds while the tab is visible and it
-- is the other person's move, so a recent one means the page is open: the move is seen,
-- and no email is sent for it (§1.5).
--
-- **`move_emails`** is one row a move email, written before it is handed to the sender.
-- `ply` is how many moves the game had when it was sent, and `(match, ply)` is unique, so
-- one move can never send two. The rows are what the site-wide daily cap counts
-- (`quota.MOVE_EMAILS_A_DAY`) and what `/admin` shows against it. `failed` is the
-- sender's refusal when there was one: nothing retries, and a failed send never touches
-- the move, which committed first.

alter table users add column move_email boolean not null default false;
alter table users add column mail_off text unique;

alter table matches add column seen0 timestamptz;
alter table matches add column seen1 timestamptz;

create table move_emails (
    id          bigint generated always as identity primary key,
    match       text not null references matches (id) on delete cascade,
    recipient   bigint not null references users (id),
    ply         integer not null,
    sent        timestamptz not null default now(),
    failed      text,
    unique (match, ply)
);

create index move_emails_sent on move_emails (sent);
