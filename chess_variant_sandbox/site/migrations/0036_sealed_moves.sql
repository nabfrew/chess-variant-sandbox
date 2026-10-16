-- A game in rounds between two people, by correspondence (Q918f,
-- `docs/plan-simultaneous.md` §6.6).
--
-- In a game in rounds (schema 26) both players choose a move for the same position in
-- secret, and both are made at once when the second is in. By correspondence the first
-- one in has to wait somewhere for the second, and nowhere a reader can reach it.
--
-- **`matches.sealed_move`** is the one move sealed in the round now open, as
-- `{"seat": 0 | 1, "move": <the move's number>, "at": <when it was sealed>}`, or null
-- while neither player has sealed. It is never part of `moves`, which is what every
-- replay, record, rating and finish reads, so nothing that rebuilds a position can show
-- it; `site/correspondence.py` is the one module that reads it, and gives the move only
-- to the seat that sealed it (the other seat learns that a move is in, never which).
-- The second seal reveals the round: under `select ... for update` on the row, both
-- moves are appended to `moves` as a pair in seat order and this is cleared, in one
-- transaction, so two seals racing cannot both find the round empty. Any finish clears
-- it too: a round never revealed stays unread.
--
-- **`move_emails`' uniqueness becomes `(match, recipient, ply)`.** A move email is still
-- one a turn: `ply` is the game's length when it was sent, which in a game in rounds is
-- the round's first ply, so a person mailed that a round opened is not mailed again that
-- the other's move is in. Two people can each be owed one at the same length in a round,
-- which `(match, ply)` refused. In a game that alternates exactly one person is to move at
-- each length, so nothing it sends changes.

alter table matches add column sealed_move jsonb;

alter table move_emails drop constraint move_emails_match_ply_key;
alter table move_emails add constraint move_emails_match_recipient_ply_key
    unique (match, recipient, ply);
