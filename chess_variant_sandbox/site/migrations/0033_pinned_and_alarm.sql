-- Which computer player sat there, exactly, and the training queue people's games may add
-- to (Q768, H4 and H9 of `docs/plan-human-data.md`).
--
-- **`matches.pinned` (H4).** A computer seat is named by its agent string, and the string
-- does not pin the player: `minimax:nodes=200000` names a search and a budget, and what that
-- search plays is also a function of the core's build, the game's compiled tables and the
-- values handed to the core (`measurement.built_fingerprint`, `tables_fingerprint`,
-- `players_fingerprint`). A learned seat's net is already pinned by `seat*_played`
-- (`0025_played.sql`); nothing pinned the rest, so a win over last month's strongest player
-- read as a win over this month's (`docs/bugs.md`). The column is a list, oldest first, of
-- each identity the row's computer moves were chosen under:
-- `{"build": <core build>, "tables": <16 hex>, "players": <16 hex or "">}`, written at the
-- start and appended by a reply only when it differs from the last entry, so one entry
-- means one player throughout and two mean a deploy changed it mid-game. Null on a row with
-- no computer seat and on every row from before this file (*unrecorded*, never *the same*).
--
-- **`jobs.cause` and `jobs.evidence` (H9).** A `train` job asks the pipeline's machine for
-- a stronger computer player on one game (`site/training.py`). `cause` says who asked:
-- `premise`, the strength-premise check's own entry, or `people`, a game over the bar of
-- `docs/plan-human-data.md` §3.3 (the user's answer B of 2026-10-08). The queue serves every
-- premise entry before any people entry. `evidence` is what a people entry was raised on:
-- the top group, the counts behind the bar and when they were read, kept so the entry can
-- be checked against the rows later. Null on every other kind of job.

alter table matches add column pinned jsonb
    check (pinned is null or jsonb_typeof(pinned) = 'array');

alter table jobs add column cause text
    check (cause is null or cause in ('premise', 'people'));
alter table jobs add column evidence jsonb
    check (evidence is null or jsonb_typeof(evidence) = 'object');
