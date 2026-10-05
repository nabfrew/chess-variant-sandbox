-- Members can send invitations (Q719, `docs/plan-launch.md` §5.3a).
--
-- **`users.invites_to_give`** is how many invitations this person may still send
-- themselves, from their own page. The maintainer sets it on `/admin`, a number, **zero by
-- default**, so nobody can invite anybody until they are given some. Sending one takes one
-- off, in the same transaction as the invitation's row and its mail, so a send that fails
-- gives it back. The invitation itself is the ordinary one (`identity.send_invitation`),
-- with the member as its issuer. Leaving clears it.

alter table users add column invites_to_give integer not null default 0
    check (invites_to_give >= 0);
