-- Schema 29 (Q983a, `docs/plan-rest.md` §4.4, §4.6): rest became a piece-type property.
--
-- Schema 27's switch `rules.simultaneous.rest: true` rested, in a game in rounds, the piece
-- that made a player's move for that player's next round. Schema 29 retired it for
-- `pieces[].rest`, a count of turns per type, and the validator and the loader refuse the old
-- key at every version, so a stored copy that still carries it would no longer build an
-- engine: a retired game's page would not draw and its matches would not replay.
--
-- **The retrofit happens once, as data, here.** Every stored spec that writes the old key is
-- rewritten to the new spelling, which is the rewrite of `docs/plan-rest.md` §4.5 and
-- `tests/rest_equivalence.py::rewrite`, step for step:
--
-- - the key goes;
-- - where it was on (`true`), `"rest": 1` goes on every piece type a player moves: one with a
--   move (a portal end has none) that no `rules.npcs` entry names (a wild piece never made a
--   player's move); for a game played in rounds only, that is the same rest move for move;
-- - `schema_version` becomes 29;
-- - a stored `provenance.history` gets the sentence the files got.
--
-- **The row keeps its hash.** It is the page's name and its matches' key, and its rules are
-- the same game spelled the new way (a spy game's alternating phase rests too under schema
-- 29, so its matches replay until a move the new rule refuses, and such a match shows its
-- record and result without the board, as a deleted game's do). The deploy's `site load`
-- then retires the old hashes that left the tree and loads the new ones, as for any in-place
-- mechanics edit. No Python reads the old key to do this.

with old as (
    select hash, spec
    from games
    where spec is not null
      and jsonb_typeof(spec -> 'rules' -> 'simultaneous') = 'object'
      and (spec -> 'rules' -> 'simultaneous') ? 'rest'
),
rested as (
    select old.hash,
           jsonb_set(
               jsonb_set(old.spec #- '{rules,simultaneous,rest}', '{schema_version}', '29'::jsonb),
               '{pieces}',
               rewritten.pieces
           ) as spec
    from old
    cross join lateral (
        select coalesce(
            jsonb_agg(
                case
                    when (old.spec -> 'rules' -> 'simultaneous' -> 'rest') = 'true'::jsonb
                         and jsonb_typeof(listed.piece -> 'moves') = 'array'
                         and jsonb_array_length(listed.piece -> 'moves') > 0
                         and not exists (
                             select 1
                             from jsonb_array_elements(
                                 case
                                     when jsonb_typeof(old.spec -> 'rules' -> 'npcs') = 'array'
                                     then old.spec -> 'rules' -> 'npcs'
                                     else '[]'::jsonb
                                 end
                             ) as npc (entry)
                             where jsonb_typeof(npc.entry) = 'object'
                               and (npc.entry -> 'piece') = (listed.piece -> 'id')
                         )
                    then listed.piece || '{"rest": 1}'::jsonb
                    else listed.piece
                end
                order by listed.position
            ),
            '[]'::jsonb
        ) as pieces
        from jsonb_array_elements(old.spec -> 'pieces') with ordinality as listed (piece, position)
    ) as rewritten
),
told as (
    select hash,
           case
               when jsonb_typeof(spec -> 'provenance' -> 'history') = 'string'
               then jsonb_set(
                   spec,
                   '{provenance,history}',
                   to_jsonb(
                       (spec -> 'provenance' ->> 'history')
                       || ' Rewritten for schema 29 by Q983a (2026-10-20): the old switch '
                       || 'rules.simultaneous.rest as "rest": 1 on every piece '
                       || '(docs/plan-rest.md section 4.5).'
                   )
               )
               else spec
           end as spec
    from rested
)
update games
set spec = told.spec
from told
where games.hash = told.hash;
