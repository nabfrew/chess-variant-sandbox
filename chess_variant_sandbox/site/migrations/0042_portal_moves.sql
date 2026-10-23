-- Every stored portal game in spelling 31: a group's moves written on the group (Q1007b,
-- `docs/plan-schema-31.md` §4.2), and every stored spec stamped 31.
--
-- Until spelling 31 a portal group named a stand-in piece declared in `pieces` for its moves
-- alone (`"piece": "O"`). Now the group carries them (`rules.portals[].moves`), and the
-- validator refuses the old spelling. This rewrites `games.spec` as
-- `spec/spelling.py::current_spelling` rewrites a file, **keeping each row's key**:
--
-- - each group that names the stand-in takes its moves as `moves` (each `"mode": "move"`
--   left out, which is what a group's move means without one; the key left out where the
--   stand-in had none), and loses `piece`;
-- - the stand-in leaves `pieces`;
-- - `schema_version` becomes 31, on every stored spec that states pieces.
--
-- **Why no name moves.** A game is named by its normal form (`spec/normal.py`), which writes
-- a group's moves back as the stand-in (`schema.portal_stand_in_shape`): a piece `{"id",
-- "name": "portal", "value": 0, "moves"}` appended last to `pieces`, its id the first of
-- `schema.PORTAL_MOVER_IDS` no other piece has, and each move's `"mode": "move"` restored. So
-- a row re-spelt here keeps its name exactly when its stand-in was that piece: every one of the
-- 59 portal specs git ever held was (`docs/reference/engine-and-spec-record.md`, *Spelling
-- 31*). The deploy's names step (`deploy/names.py`) compares stored specs before migrations
-- run, so it cannot see a rename made here; this file guards itself instead. **A stored portal
-- game of any other shape stops the migration**, naming its key, before anything is written
-- (the transaction rolls back): one stand-in for every group, last in `pieces`, with no key
-- but `id`, `name` `"portal"`, `value` 0 and `moves` (and one the normal form drops), each
-- move `"mode": "move"`, and the id the normal form would give it. The cure is a legacy-table entry for the row's re-spelt
-- spec, as `deploy/README.md` *The names of stored games* gives for a refused name, and a
-- deploy of that table with this file taught the row.
--
-- A deleted game's row has no spec and a row that states no pieces is no game; both are left
-- alone. Running the file twice changes nothing. `tests/test_site_portal_moves.py` holds it to
-- the Python converter row for row against a scratch store. The functions are temporary.

-- The id `schema.portal_movement_types` gives the movement type of a spec whose other pieces
-- are `pieces`: the first of `schema.PORTAL_MOVER_IDS` none of them has.
create function pg_temp.portal_moves_mover_id(pieces jsonb) returns text
language sql immutable as $$
    select letter
    from unnest(string_to_array('O,V,W,X,Y,Z,J,L,M,T,U,Q,P,R,S,A,B,C,D,E,F,G,H,I,K,N,0,1,2,3,4,5,6,7,8,9', ','))
         with ordinality as candidate(letter, place)
    where not exists (
        select 1 from jsonb_array_elements(pieces) piece where piece ->> 'id' = letter
    )
    order by place
    limit 1
$$;

-- Whether a stored spec's portal groups name a stand-in of the one shape the normal form
-- writes back. Called only on a spec some group of which names a `piece`.
create function pg_temp.portal_moves_one_shape(spec jsonb) returns boolean
language plpgsql immutable as $$
declare
    groups jsonb := spec -> 'rules' -> 'portals';
    pieces jsonb := spec -> 'pieces';
    named text;
    stand jsonb;
    last integer;
begin
    if jsonb_typeof(pieces) <> 'array' or jsonb_array_length(pieces) < 2 then
        return false;
    end if;
    -- Every group an object naming one and the same piece.
    if exists (
        select 1 from jsonb_array_elements(groups) g
        where jsonb_typeof(g) <> 'object' or jsonb_typeof(g -> 'piece') <> 'string'
    ) or (select count(distinct g ->> 'piece') from jsonb_array_elements(groups) g) <> 1 then
        return false;
    end if;
    named := groups -> 0 ->> 'piece';
    if (select count(*) from jsonb_array_elements(pieces) p where p ->> 'id' = named) <> 1 then
        return false;
    end if;
    last := jsonb_array_length(pieces) - 1;
    stand := pieces -> last;
    if stand ->> 'id' is distinct from named then
        return false;
    end if;
    -- Beside `id`, `name`, `value` and `moves`, only a key the normal form drops: one at its
    -- default (`royal: false`) or at its off value (`schema.PIECE_OFF_VALUES`).
    if exists (
        select 1 from jsonb_each(stand) kv
        where kv.key not in ('id', 'name', 'value', 'moves')
          and (kv.key, kv.value) not in (('royal', 'false'::jsonb), ('rest', '0'::jsonb),
                                         ('speed', '0'::jsonb), ('spy', 'false'::jsonb),
                                         ('sight', '[]'::jsonb))
    ) or stand -> 'name' is distinct from '"portal"'::jsonb
      or (stand -> 'value')::text is distinct from '0'
      or jsonb_typeof(stand -> 'moves') is distinct from 'array' then
        return false;
    end if;
    if exists (
        select 1 from jsonb_array_elements(stand -> 'moves') m
        where jsonb_typeof(m) <> 'object' or m -> 'mode' is distinct from '"move"'::jsonb
    ) then
        return false;
    end if;
    return named = pg_temp.portal_moves_mover_id(pieces - last);
end
$$;

-- `spelling._portal_moves_on_groups`: the groups with the stand-in's moves, and no stand-in.
create function pg_temp.portal_moves_respelt(spec jsonb) returns jsonb
language plpgsql immutable as $$
declare
    named text := spec -> 'rules' -> 'portals' -> 0 ->> 'piece';
    moves jsonb;
    groups jsonb;
begin
    select coalesce(jsonb_agg(m - 'mode' order by place), '[]'::jsonb) into moves
    from jsonb_array_elements(
        (select p -> 'moves' from jsonb_array_elements(spec -> 'pieces') p
         where p ->> 'id' = named)
    ) with ordinality as listed(m, place);
    select jsonb_agg(
        case when moves = '[]'::jsonb then g - 'piece'
             else (g - 'piece') || jsonb_build_object('moves', moves) end
        order by place
    ) into groups
    from jsonb_array_elements(spec -> 'rules' -> 'portals') with ordinality as listed(g, place);
    return jsonb_set(
        jsonb_set(spec, '{rules,portals}', groups),
        '{pieces}',
        (select jsonb_agg(p order by place)
         from jsonb_array_elements(spec -> 'pieces') with ordinality as listed(p, place)
         where p ->> 'id' is distinct from named)
    );
end
$$;

do $$
declare
    odd text;
begin
    select string_agg(hash, ', ' order by hash) into odd
    from games
    where jsonb_typeof(spec) = 'object'
      and jsonb_typeof(spec -> 'rules' -> 'portals') = 'array'
      and exists (
          select 1 from jsonb_array_elements(spec -> 'rules' -> 'portals') g
          where jsonb_typeof(g) = 'object' and g ? 'piece'
      )
      and not pg_temp.portal_moves_one_shape(spec);
    if odd is not null then
        raise exception 'stored portal games of another shape, which re-spelling would rename (0042_portal_moves.sql says the cure): %', odd;
    end if;
end
$$;

update games
set spec = pg_temp.portal_moves_respelt(spec)
where jsonb_typeof(spec) = 'object'
  and jsonb_typeof(spec -> 'rules' -> 'portals') = 'array'
  and exists (
      select 1 from jsonb_array_elements(spec -> 'rules' -> 'portals') g
      where jsonb_typeof(g) = 'object' and g ? 'piece'
  );

update games
set spec = spec || '{"schema_version": 31}'
where jsonb_typeof(spec) = 'object'
  and spec ? 'pieces';

drop function pg_temp.portal_moves_respelt(jsonb);
drop function pg_temp.portal_moves_one_shape(jsonb);
drop function pg_temp.portal_moves_mover_id(jsonb);
