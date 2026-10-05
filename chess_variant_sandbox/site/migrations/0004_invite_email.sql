-- The address an invitation was sent to, so the link is one person's.
--
-- Task 1.3. An invitation goes by email (§10, decided 2026-09-13), so the code in the
-- link belongs to the address it was sent to and the person who follows it chooses a
-- handle and a password and nothing else. Without this column the join form would have
-- to ask for an address, and any address, which is a different invitation.
--
-- Nullable, because `uses_left` allows a code meant for more than one person and
-- `issuer` is already null for the first invite. `newgame site invite` always fills it;
-- a code without one is refused at the join, in a sentence, rather than quietly letting
-- somebody name their own address.

alter table invites add column email text;

-- One handle, whatever its capitals. `users.handle` is unique already, exactly, which
-- would let `ada` and `Ada` be two people; sign-in matches case-insensitively, so they
-- would be two people one password could reach. The index is what makes the rule the
-- database's rather than a query's.

create unique index users_one_handle on users (lower(handle));
