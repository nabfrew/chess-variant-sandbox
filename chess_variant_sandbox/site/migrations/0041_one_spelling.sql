-- Every stored spec in the one current spelling (Q1008b,
-- `docs/reference/schema-reboot-assessment.md` §5 step 3).
--
-- Since Q1008a a game is named by its normal form (`spec/normal.py`), or by the name it had
-- before, through `spec/legacy_names.py`; the spelling of its file no longer names it. So the
-- files under `rulesets/` and `archive/` were rewritten once into one spelling by
-- `spec/spelling.py::current_spelling`, and this rewrites `games.spec` the same way, row by
-- row, **keeping each row's key**. Every rewrite below is one the normal form also makes, so
-- each row is named after it as before it: its key, or, for the three rows `0039` re-spelt in
-- place, the name they already had apart from their key (`deploy/names.py`).
--
-- The rewrites, step for step those of `current_spelling` (whose docstring says why each is
-- one spelling), held to it row for row by `tests/test_site_one_spelling.py`:
--
-- - `schema_version` becomes 30;
-- - a rule stated at the value that means no rule goes: one of the rules that are nothing
--   when empty at `false`, `null`, `0`, `""`, `[]` or `{}`; a `turn` of one move; a `wrap`
--   whose every edge is `"none"`; a `strike` that is plain capture;
-- - `promotion` is one object when it has one entry, a list when it has several; in each
--   entry a numbered zone is `ranks`, ascending and without repeats (a lone `rank` on the
--   board becomes a one-rank `ranks`; one past the board's edge stays), and `to_by_player`
--   naming one list for both players is `to`;
-- - a `wrap` is written key by key for the board's shape, `X_by_player` only where the two
--   players differ.
--
-- A setting written at its default, `"rank": "last"`, `strike` beside `contacts` and a
-- portal pair beside an `ends` group stay as written, as in the files. Names, descriptions,
-- appearance and provenance are not touched. A deleted game's row has no spec, and a row
-- whose spec states no pieces is no game; both are left alone. The functions are temporary:
-- they are dropped before the transaction commits.

create function pg_temp.one_spelling_falsy(value jsonb) returns boolean
language sql immutable as $$
    select value is null or value in ('null', 'false', '0', '""', '[]', '{}')
$$;

create function pg_temp.one_spelling_int(value jsonb) returns boolean
language sql immutable as $$
    select coalesce(jsonb_typeof(value) = 'number' and value::text ~ '^-?[0-9]+$', false)
$$;

-- `normal._normal_wrap`: each of the shape's keys written once, or per player where the two
-- differ; a wrap with a key the shape does not have, or an edge that is not a word, as it was.
create function pg_temp.one_spelling_wrap(wrap jsonb, shape jsonb) returns jsonb
language plpgsql immutable as $$
declare
    axes text[] := case when shape = '"hexagon"' then array['n_s', 'ne_sw', 'nw_se']
                        else array['files', 'ranks'] end;
    axis text;
    shared jsonb;
    per jsonb;
    first jsonb;
    second jsonb;
    written jsonb := '{}';
begin
    if exists (
        select 1 from jsonb_object_keys(wrap) as key
        where not key = any (axes || array(select a || '_by_player' from unnest(axes) a))
    ) then
        return wrap;
    end if;
    foreach axis in array axes loop
        shared := coalesce(wrap -> axis, '"none"');
        per := wrap -> (axis || '_by_player');
        if jsonb_typeof(per) = 'object' and per <> '{}' then
            first := coalesce(per -> '0', shared);
            second := coalesce(per -> '1', shared);
        else
            first := shared;
            second := shared;
        end if;
        if jsonb_typeof(first) <> 'string' or jsonb_typeof(second) <> 'string' then
            return wrap;
        end if;
        if first = second then
            if first <> '"none"' then
                written := written || jsonb_build_object(axis, first);
            end if;
        else
            written := written || jsonb_build_object(
                axis || '_by_player', jsonb_build_object('0', first, '1', second)
            );
        end if;
    end loop;
    return written;
end
$$;

-- `spelling._promotion`: one entry.
create function pg_temp.one_spelling_entry(entry jsonb, height jsonb) returns jsonb
language plpgsql immutable as $$
declare
    zoned boolean := not (entry ? 'squares') and not (entry ? 'rank' and entry ? 'ranks');
    by_player jsonb := entry -> 'to_by_player';
    out jsonb := entry;
begin
    if not (entry ? 'to')
       and jsonb_typeof(by_player) = 'object'
       and (select array_agg(key order by key) from jsonb_object_keys(by_player) key)
           = array['0', '1']
       and by_player -> '0' = by_player -> '1' then
        out := (out - 'to_by_player') || jsonb_build_object('to', by_player -> '0');
    end if;
    if zoned and pg_temp.one_spelling_int(entry -> 'rank') and pg_temp.one_spelling_int(height)
       and (entry ->> 'rank')::numeric >= 0
       and (entry ->> 'rank')::numeric < (height #>> '{}')::numeric then
        out := (out - 'rank') || jsonb_build_object('ranks', jsonb_build_array(entry -> 'rank'));
    end if;
    if zoned and jsonb_typeof(entry -> 'ranks') = 'array'
       and not exists (
           select 1 from jsonb_array_elements(entry -> 'ranks') r
           where not pg_temp.one_spelling_int(r)
       ) then
        out := out || jsonb_build_object('ranks', (
            select coalesce(jsonb_agg(r order by r::text::numeric), '[]')
            from (select distinct r from jsonb_array_elements(entry -> 'ranks') r) as d
        ));
    end if;
    return out;
end
$$;

-- `spelling._rules`: each rule in its one spelling, and each stated-off rule left out.
create function pg_temp.one_spelling_rules(rules jsonb, height jsonb, shape jsonb)
returns jsonb
language plpgsql immutable as $$
declare
    key text;
    value jsonb;
    entries jsonb;
    out jsonb := '{}';
begin
    for key, value in select * from jsonb_each(rules) loop
        if key = 'promotion' and jsonb_typeof(value) in ('object', 'array') then
            entries := case when jsonb_typeof(value) = 'object'
                            then jsonb_build_array(value) else value end;
            select coalesce(jsonb_agg(
                       case when jsonb_typeof(e) = 'object'
                            then pg_temp.one_spelling_entry(e, height) else e end
                       order by i), '[]')
              into entries
              from jsonb_array_elements(entries) with ordinality as listed (e, i);
            -- Asked of the list, before one entry is unwrapped: only `[]` states nothing.
            if entries <> '[]' then
                out := out || jsonb_build_object(key, case
                    when jsonb_array_length(entries) = 1 and jsonb_typeof(entries -> 0) = 'object'
                    then entries -> 0 else entries end);
            end if;
            continue;
        elsif key = 'wrap' and jsonb_typeof(value) = 'object' then
            value := pg_temp.one_spelling_wrap(value, shape);
        end if;
        -- A turn of one move: only `moves` and `ramp`, one move, every ramp entry one.
        if key = 'turn' and jsonb_typeof(value) = 'object'
           and not exists (
               select 1 from jsonb_object_keys(value) k where k not in ('moves', 'ramp')
           )
           and coalesce(value -> 'moves', '1') = '1'
           and (pg_temp.one_spelling_falsy(value -> 'ramp')
                or (jsonb_typeof(value -> 'ramp') = 'array'
                    and not exists (
                        select 1 from jsonb_array_elements(value -> 'ramp') r where r <> '1'
                    ))) then
            continue;
        end if;
        -- A strike that is plain capture (`schema.strike_is_live` says no).
        if key = 'strike' and jsonb_typeof(value) = 'object'
           and coalesce(value -> 'effect', '"capture"') = '"capture"'
           and coalesce(value -> 'position', '"advance"') not in ('"stay"', '"home"')
           and pg_temp.one_spelling_falsy(value -> 'switches_sides') then
            continue;
        end if;
        -- `normal._ABSENT_WHEN_EMPTY` at one of `normal._EMPTY`.
        if key in ('promotion', 'en_passant', 'castling', 'bare_royal', 'check_limit', 'reach',
                   'extinction', 'must_capture', 'forbid_check', 'drops',
                   'explode_on_capture', 'forbid_immobility', 'scoring', 'territory', 'turn',
                   'wrap', 'obstacles', 'strike', 'contacts', 'npcs', 'promote_on_entry',
                   'portals')
           and value in ('null', 'false', '0', '""', '[]', '{}') then
            continue;
        end if;
        out := out || jsonb_build_object(key, value);
    end loop;
    return out;
end
$$;

update games
set spec = (
    case when jsonb_typeof(spec -> 'rules') = 'object'
         then jsonb_set(
             spec,
             '{rules}',
             pg_temp.one_spelling_rules(
                 spec -> 'rules',
                 case when jsonb_typeof(spec -> 'board') = 'object'
                      then spec -> 'board' -> 'height' end,
                 case when jsonb_typeof(spec -> 'board') = 'object'
                      then spec -> 'board' -> 'shape' end
             )
         )
         else spec end
) || '{"schema_version": 30}'
where jsonb_typeof(spec) = 'object'
  and spec ? 'pieces';

drop function pg_temp.one_spelling_rules(jsonb, jsonb, jsonb);
drop function pg_temp.one_spelling_entry(jsonb, jsonb);
drop function pg_temp.one_spelling_wrap(jsonb, jsonb);
drop function pg_temp.one_spelling_int(jsonb);
drop function pg_temp.one_spelling_falsy(jsonb);
