-- How far the machine looked before each move, beside the move.
--
-- Q424, after Q288. The board shows *looked 3 moves ahead* beside a searched machine
-- move, and the number is the core's (`Engine.select_move_depth`), reported once, with
-- the reply that played it. The site rebuilds a match from `moves` on every request, and
-- a replayed move is not searched again, so a reload lost every depth the page had been
-- told. This column keeps them, so a reloaded board shows what it showed before.
--
-- **`depths` runs beside `moves`, one entry per move, in the same order.** An entry is
-- the depth as an integer, or null for a move with nothing to report: a person's, a
-- player that does not deepen (random, greedy, the playout search), and a position with
-- one legal move, where no search happens (`web/play.py::_reply`). It is a second array
-- rather than objects inside `moves` because `moves` is what replays the game, and every
-- reading of it -- `jsonb_array_length(moves)` for the ply count among them -- stays as
-- it was.
--
-- **Null means the row never recorded any.** A row written before this migration has
-- none, and loads as it always did, with no depth beside any move. Its next move writes
-- the array: null for every move before it, and whatever the core reported for the new
-- one. A match starts null too and gains the array with its first move. No backfill:
-- the depths were never kept, and a search cannot be run again to find them.
--
-- **The constraint keeps the two arrays the same length.** `site/matches.py` writes
-- both in one statement, and a writer that changed `moves` alone would leave every depth
-- beside the wrong move without a word. The `case` is there because Postgres does not
-- promise to evaluate an `and` left to right, and `jsonb_array_length` raises on a value
-- that is not an array rather than returning false.

alter table matches add column depths jsonb;

alter table matches add constraint matches_depths_beside_moves check (
    case
        when depths is null then true
        when jsonb_typeof(depths) <> 'array' then false
        else jsonb_array_length(depths) = jsonb_array_length(moves)
    end
);
