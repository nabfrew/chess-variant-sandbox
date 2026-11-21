-- A member may change their display name (DN2, `docs/reference/site-as-built.md` §6).
--
-- **`former_handles`** keeps every name a member has given up, so that nobody else can take
-- it (a name once used is never free for someone else: that would let a newcomer pass for
-- the previous holder), the member can go back to one of their own, and `/players/<old>`
-- still finds them. Unique ignoring case, as `users_one_handle` is; `identity._free` checks
-- both tables, so a current name and a former one can never be held by two people. A row is
-- deleted when its member takes the name back, and every row of a member is deleted when they
-- leave (`admin.forget`), because a name is personal data the privacy notice says goes.
--
-- **`users.handle_changed`** is when the member last changed their name, null for one who
-- never has: the limit (`identity.RENAME_EVERY`) reads it, and the first change is free.
--
-- Safe on a store that already has members: one new empty table and one nullable column, so
-- every existing member starts with no former names and their first change still to make.

create table former_handles (
    handle text not null,
    user_id bigint not null references users (id) on delete cascade,
    until timestamptz not null default now()
);

create unique index former_handles_one_name on former_handles (lower(handle));
create index former_handles_by_user on former_handles (user_id);

alter table users add column handle_changed timestamptz;
