-- The top group a leaderboard match was started against, beside the bit that says
-- whether this match's own opponent was in it.
--
-- §5 of docs/plan-website.md asks for two things at once. A clear is "decided at the
-- time" and permanent, so it cannot be re-read off a table human play keeps moving; and
-- "clearing a tied game means beating each of the tied", so the decision needs the whole
-- tied group and not one yes-or-no about one opponent. Task 1.5 found that one boolean
-- cannot carry both: with `top` alone, a win over one of two tied computer players is
-- indistinguishable from a win over a lone top, and the tag either clears too early or
-- loses its permanence when the top changes hands.
--
-- So the row carries the group. `top` keeps exactly the meaning task 1.4 gave it and is
-- now also `agent = any (top_players)`; `top_players` is every computer player the
-- game's table read as top when this match started, in rating order. A sandbox row, a
-- row between two people and a row loaded from the local application all leave it empty,
-- which is what an empty array means here: this match was never played against a top.

alter table matches add column top_players text[] not null default '{}';

-- No index. The tag reads a game's leaderboard rows in finish order, which
-- `matches_by_game` already covers, and this column is read off rows that query returns.
