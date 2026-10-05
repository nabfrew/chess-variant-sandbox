-- The maker takes challengers (Q606, `docs/plan-return.md` §3 C4 and R10).
--
-- **`users.takes_challengers`** is one switch on a person's own page, **off by default**:
-- *list me on the games I made as taking challengers*. Switched on, the person is named
-- first in *Play a person* on each of their published designed games, as *‹handle› made
-- this game and takes challengers*, and any other member may challenge them there through
-- the ordinary challenge (no message of any kind). It is refused to somebody with no
-- published game of their own, never lapses, and sends nothing to anybody. Switching it
-- off always works. Leaving and being revoked clear it.

alter table users add column takes_challengers boolean not null default false;
