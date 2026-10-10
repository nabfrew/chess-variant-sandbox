-- What people's games can tell us: think time, what a take-back undid, and the rating at
-- the time on a practice game (Q767, H1 to H3 of `docs/plan-human-data.md` §1.2).
--
-- None of it is shown per person, and none of it reaches a score (rule 5 of
-- `docs/plan-website.md`): it is read with the game's other moves, by the export
-- (`chess-variant-sandbox site games`), and by nothing that rates, measures or sorts.
--
-- **`matches.thinks` (H1)** runs beside `moves`, one entry per move, in the same order, the
-- way `depths` does (`0014_depths.sql`): the milliseconds between the server storing the
-- computer player's reply (or the start, or a take-back) and the person's move arriving,
-- measured on the database's clock. Null for a computer player's move, and null for a gap
-- of more than 30 minutes (*away*, not *thinking*; the cap is the plan's, fixed before any
-- gap was read). Written only on a game between a person and a computer player: a game
-- between two people keeps `times` and gains nothing, and a watch has nobody thinking.
-- Null on every other row and on every row from before this migration, whose next move
-- writes the array with null for every earlier move. The constraint holds it to the
-- length of `moves`, for `depths`' reason.
--
-- **`matches.undone` (H2)** is a list that each take-back and restart appends to, oldest
-- first: `{"ply": n, "moves": [...], "thinks": [...]}`, the ply the board went back to and
-- the moves it removed (with their think times, where the row kept them); a restart's
-- entry also says `"restart": true`. It is never emptied, so the take-backs of a whole
-- practice row are one list, and the position each went back to can be rebuilt backwards
-- from the moves that followed. Null until the first take-back.
--
-- **`matches.standing` (H3)** is each side's rating and deviation in the game's table when
-- a practice game against a computer player started (or was restarted), on a game whose
-- table has computer players: `{"ratings": [r0, r1], "deviations": [d0, d1],
-- "starting": [b0, b1]}`. `starting` is true for a person with no row on that table yet,
-- whose values are the table's starting ones. A seat with nothing to read -- a visitor,
-- a computer player the table does not hold -- is null in each list. A read only: a
-- practice game still moves nothing. A rated game keeps its `seat*_rating_before`, written
-- at the finish, and leaves this null.
--
-- **`finishes` copies all three** at a finish, as it copies `takebacks`, because a
-- take-back or a restart rewrites the match row afterwards.

alter table matches add column thinks jsonb;

alter table matches add constraint matches_thinks_beside_moves check (
    case
        when thinks is null then true
        when jsonb_typeof(thinks) <> 'array' then false
        else jsonb_array_length(thinks) = jsonb_array_length(moves)
    end
);

alter table matches add column undone jsonb
    check (undone is null or jsonb_typeof(undone) = 'array');

alter table matches add column standing jsonb
    check (standing is null or jsonb_typeof(standing) = 'object');

alter table finishes add column thinks jsonb
    check (thinks is null or jsonb_typeof(thinks) = 'array');
alter table finishes add column undone jsonb
    check (undone is null or jsonb_typeof(undone) = 'array');
alter table finishes add column standing jsonb
    check (standing is null or jsonb_typeof(standing) = 'object');
