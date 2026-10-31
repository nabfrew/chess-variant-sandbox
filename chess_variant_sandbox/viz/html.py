"""Render a recorded game as a single self-contained HTML file.

No bundler, no CDN, no image assets: the whole page is one file that opens from disk.
That is a deliberate constraint for this project rather than minimalism for its own
sake -- a sweep produces thousands of games across dozens of rulesets, and the useful
thing to be able to do with an interesting one is send the file to someone.

Pieces are drawn from ``assets/pieces`` where the library has an icon for the name,
and as their spec letters where it does not -- which is the ordinary case for an
invented piece. The icons are inlined as data URIs, so the page is still one file.
Pass ``icons=False`` for letters throughout: a borrowed glyph quietly implies that a
piece moves like the chess piece it resembles, which is worth suppressing when the
point of the replay is a proposal rather than a known game.

**A step is a beat, not a ply.** The replay has no animation layer: it redraws the
position and marks the two squares that changed. In a game with wild pieces that was a
ply drawn as one change -- the mover's move and the phase the rules run after it inside
the same ``make`` -- so four wolves moved at once with nothing marking where any of them
came from, and a wolf's capture looked like a piece vanishing. Each ply's frame now
carries the core's own log of the phase (``Engine.apply_with_phase``), and the page
splits it with ``plyBeats``, the play board's split, shared through ``viz/wrapped.py``:
one step stands between the move and the phase, the marks are the squares that beat
changed, and the move list gives the phase a row worded by ``phaseWords``, the same
words the play board's list uses. A game without wild pieces has one beat per ply and
replays exactly as it did.

Two moves are drawn moving because their marks cannot explain them: a crossing of a seam
(`showCrossing`), and since Q1074bo a push (`showPush`), whose line the frame carries from
the engine (`viz.wrapped.push_transit`): the line slides one step on, and the piece it threw
off falls over the edge, or the piece it crushed on a wall is pressed flat where it stood.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.moves import (
    DUCK_PASS,
    DUCK_PLACE,
    MOVE_DUCK,
    move_promotion,
    move_type,
)
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.tournament.play import replay
from chess_variant_sandbox.tournament.record import GameRecord
from chess_variant_sandbox.viz.pieces import credits_for, icons_for, side_names
from chess_variant_sandbox.viz.rules_payload import rules_payload
from chess_variant_sandbox.viz.wrapped import (
    WRAPPED,
    fold_payload,
    fold_transit,
    portal_transit,
    push_transit,
)


def _duck_squares(move: int) -> dict[str, Any]:
    """A duck move's squares to mark (generation 42): none of the pass's, only a placement's
    target; empty for every other move, which keeps its own."""
    if move_type(move) != MOVE_DUCK:
        return {}
    how = move_promotion(move)
    if how == DUCK_PASS:
        return {"from": None, "to": None}
    return {"from": None} if how == DUCK_PLACE else {}


def _purse(state: GameState) -> list[int]:
    """Each player's coins in ``state``, as the core keeps them (``(0, 0)`` without a
    purse); the replay's counterpart of the play board's ``purse`` (`web/play.py`)."""
    return [int(state.purse[0]), int(state.purse[1])]


def render_game(
    spec: dict[str, Any], record: GameRecord, out_path: str | Path, icons: bool = True
) -> Path:
    """Write an interactive replay of ``record`` to ``out_path``."""
    engine = Engine(spec)
    states = replay(engine, record)

    frames = []
    state = engine.initial_state()
    for index, move in enumerate(record.moves):
        after, phase = engine.apply_with_phase(state, move)
        frames.append(
            {
                "text": engine.describe_move(state, move),
                "from": move & 0xFF,
                "to": (move >> 8) & 0xFF,
                # Generation 42 (N4c): a duck placement has no origin to mark, and the pass
                # no square at all; the text says each (`describe_move`).
                **_duck_squares(move),
                # What the wild-piece phase did after this move (schema 9), as the core
                # logged it: one ``{from, to, eaten}`` per wild piece that walked. The
                # page splits the ply into beats from it; empty for every ruleset
                # without ``rules.npcs``.
                "phase": [step.as_dict() for step in phase],
                # Who made it, which is whose passage a portal shows on the position
                # before it; and, for a move through a portal (schema 19), the two ends
                # it passed, so the replay marks both (`viz.wrapped.portal_transit`).
                "player": state.side_to_move,
                **(portal_transit(engine, state, move) or {}),
                # Schema 22 (Q843d): a move through a fold, its way in and out, marked too.
                **(fold_transit(engine, state, move) or {}),
                # Q1074bo: a push's line, the play board's own (`viz.wrapped.push_transit`):
                # each piece it shoved and where to, the step, and whether the last piece
                # was crushed on a wall rather than thrown off, so the replay slides the
                # line and drops or crushes the last piece as the play board does.
                **(push_transit(engine, state, move) or {}),
                "material": record.material[index] if index < len(record.material) else None,
                "branching": record.branching[index] if index < len(record.branching) else None,
            }
        )
        state = after

    return _write_page(
        engine,
        spec,
        boards=[list(s.board) for s in states],
        folds=[fold_payload(engine, s) for s in states],
        purses=[_purse(s) for s in states],
        frames=frames,
        meta={
            "agents": list(record.agents),
            "seed": record.seed,
            "winner": record.winner,
            "reason": record.reason,
            "plies": record.plies,
        },
        out_path=out_path,
        icons=icons,
    )


def render_position(
    spec: dict[str, Any], state: GameState, out_path: str | Path, icons: bool = True
) -> Path:
    """Write a single position. Useful when a smoke test rejects a ruleset.

    Looking at the position a proposal gets stuck in is usually faster than reading its
    rules and working out why.
    """
    engine = Engine(spec)
    return _write_page(
        engine,
        spec,
        boards=[list(state.board)],
        folds=[fold_payload(engine, state)],
        purses=[_purse(state)],
        frames=[],
        meta={
            "agents": ["-", "-"],
            "seed": 0,
            "winner": None,
            "reason": f"position at ply {state.ply}",
            "plies": state.ply,
        },
        out_path=out_path,
        icons=icons,
    )


def _write_page(
    engine: Engine,
    spec: dict[str, Any],
    boards: list[list[int]],
    frames: list[dict[str, Any]],
    meta: dict[str, Any],
    out_path: str | Path,
    icons: bool = True,
    folds: list[list[dict[str, Any]]] | None = None,
    purses: list[list[int]] | None = None,
) -> Path:
    art = (
        icons_for(
            engine.spec.piece_names,
            engine.spec.piece_ids,
            (spec.get("appearance") or {}).get("icons"),
        )
        if icons
        else []
    )
    payload = {
        # Every compiled rule a page draws (`viz.rules_payload`, CP5), the play board's
        # block, so the replay carries what the board does: walls, wrap, goals, zones,
        # portals, the blast, speeds, rests and the purse among them.
        **rules_payload(engine),
        "pieceIcons": art,
        "credits": credits_for(art),
        "boards": boards,
        # Schema 22 (Q843d): where the folds lie in each position, no board byte, so the replay
        # draws each pool under the piece standing in it. Absent without folds.
        **({"folds": folds} if folds and any(folds) else {}),
        # TC7 (Q1049): each player's coins in each position, one pair a ply, read off the
        # core's replayed state (`GameState.purse`), never worked out on the page; with
        # the names the seats go by, which the purses sit under as on the play board.
        # Absent without a purse.
        **(
            {"purses": purses, "sides": list(side_names(art))}
            if purses and engine.spec.purse is not None
            else {}
        ),
        "moves": frames,
        "meta": {"ruleset": spec.get("name", "?"), "hash": spec_hash(spec), **meta},
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_PAGE.replace("__DATA__", json.dumps(payload)), encoding="utf-8")
    return out_path


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chess variant sandbox replay</title>
<style>
  :root {
    --bg: #f6f5f2; --panel: #ffffff; --ink: #1b1c1e; --muted: #6b6f76;
    --line: #d9d6d0; --light: #ece6da; --dark: #b6a894; --wall: #6f6558;
    --p0: #ffffff; --p0-ink: #16171a; --p1: #26282c; --p1-ink: #f4f2ee;
    /* A third owner (schema 9): a wild piece is neither player's, so it is drawn
       in neither player's colour. Ochre because the two armies are a light/dark
       pair whatever the theme, and a third piece has to read as *off* that axis
       rather than as a shade between them. It never takes an army's icon -- an
       icon pair has one drawing per player and no third entry -- so a wild piece
       is always the lettered disc. */
    --p2: #c08a3e; --p2-ink: #16171a;
    --mark: #d9a441;
    /* A square the rules end the game on when it is reached. Teal because amber
       is already the last move's and ochre already a wild piece's, and a marker
       that borrowed either would read as that thing in the wrong place. */
    --goal: #1f8f86;
    /* A seam a piece may cross one way and not the other, and the squares a piece
       promotes on. Blue and rose because the board's other colours are spoken
       for: amber, ochre, teal, and green and purple for ground held. */
    --seam: #3466ac; --zone: #b0468a;
    /* A portal end (schema 19, drawn as Q838b's vortex): each pair's colour, the same
       at both its ends and in both themes, the near-black that edges the well and falls
       into its eye, and the stone of the designer's pending end; the play board's own
       values. */
    --pair-1: #7b4fd6; --pair-2: #efc12b; --pair-3: #22b8cf; --pair-4: #d55e00;
    --portal-edge: #1e140c; --portal-stone: #b3aca1;
    /* A fold portal's pool, lit near white in its middle (schema 22, Q843d). */
    --fold-light: #fffbe8;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #131416; --panel: #1c1e21; --ink: #eceae6; --muted: #9aa0a8;
      --line: #2e3136; --light: #4a4640; --dark: #2f2c28; --wall: #16150f;
      --p0: #f0ede7; --p0-ink: #16171a; --p1: #3b3f45; --p1-ink: #f4f2ee;
      --p2: #8f6a2f; --p2-ink: #f4f2ee;
      --goal: #5cc9bd;
      --seam: #74a6ef; --zone: #e58cc4;
      --portal-stone: #9b948a;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px; background: var(--bg); color: var(--ink);
    font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }
  .wrap { max-width: 1000px; margin: 0 auto; display: grid; gap: 20px;
          grid-template-columns: minmax(280px, 1fr) 260px; }
  @media (max-width: 760px) { .wrap { grid-template-columns: 1fr; } }
  h1 { font-size: 18px; margin: 0 0 2px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 14px; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
  svg { width: 100%; height: auto; display: block; }
  .board-frame { overflow: hidden; border-radius: 4px; }
  .controls { display: flex; gap: 6px; margin-top: 12px; flex-wrap: wrap; align-items: center; }
  .controls .muted { color: var(--muted); font-size: 13px; }
  button {
    font: inherit; padding: 6px 12px; border-radius: 7px; cursor: pointer;
    border: 1px solid var(--line); background: var(--panel); color: var(--ink);
  }
  button:hover { border-color: var(--muted); }
  button:disabled { opacity: .4; cursor: default; }
  .moves { max-height: 60vh; overflow-y: auto; margin-top: 10px; }
  .move { padding: 3px 8px; border-radius: 5px; cursor: pointer; font-variant-numeric: tabular-nums; }
  .move:hover { background: var(--bg); }
  /* A wild-piece phase, under the move it answered. */
  .move.phase { padding-left: 24px; color: var(--muted); font-size: 13px; }
  .move.active { background: var(--mark); color: #16171a; }
  .stat { display: flex; justify-content: space-between; gap: 10px; font-size: 13px; padding: 2px 0; }
  .stat span:first-child { color: var(--muted); }
  /* The seats and their purses (TC7), as the play board's seat bar words them. */
  .seats { display: grid; gap: 6px; margin-bottom: 8px; padding-bottom: 8px;
           border-bottom: 1px solid var(--line); font-size: 13px; }
  .seats .muted { color: var(--muted); }
  .credits { color: var(--muted); font-size: 11px; margin-top: 14px; }
  .credits div { padding: 1px 0; }
  /* A portal's arms turn idly inwards, as on the play board (the page's `.pg-idle`), and
     not at all under reduced motion; an end nobody may go in by has no arms to turn. The
     replay draws the glyph whole: its beak outside the well, its arms by their edges. */
  .pg-wedge, .pg-mass, .pg-beak-cut { display: none; }
  .pg-idle { animation: pg-turn 14s linear infinite; }
  @keyframes pg-turn { to { transform: rotate(-360deg); } }
  @media (prefers-reduced-motion: reduce) { .pg-idle { animation: none; } }
</style>
</head>
<body>
<div class="wrap">
  <div>
    <h1 id="title"></h1>
    <div class="sub" id="subtitle"></div>
    <div class="panel"><div class="board-frame"><svg id="board" viewBox="0 0 100 100"></svg></div></div>
    <div class="controls">
      <button id="start">&#124;&#9664;</button>
      <button id="prev">&#9664;</button>
      <button id="play">&#9654; play</button>
      <button id="next">&#9654;</button>
      <button id="end">&#9654;&#124;</button>
    </div>
    <div class="controls" id="shift"></div>
    <div class="controls" id="marks"></div>
    <div class="credits" id="credits"></div>
  </div>
  <div class="panel">
    <div class="seats" id="seats" hidden></div>
    <div class="stat"><span>ply</span><span id="ply"></span></div>
    <div class="stat"><span>material</span><span id="material"></span></div>
    <div class="stat"><span>legal moves</span><span id="branching"></span></div>
    <div class="moves" id="moves"></div>
  </div>
</div>
<script>
__WRAPPED__
const DATA = __DATA__;
__REPLAY__</script>
</body>
</html>
"""

# The replay's own script is `replay.js` beside this module, and the wrapped board is
# `wrapped.js`, shared with the play board rather than written twice. The replay is a
# single file that opens from disk, so both are inlined here, and the game goes in as
# `DATA` when a file is written. Nothing serves it: the page a server sends carries no
# script body (Q513), and this is a file, not an answer.
_PAGE = _TEMPLATE.replace("__WRAPPED__", WRAPPED).replace(
    "__REPLAY__", Path(__file__).with_name("replay.js").read_text(encoding="utf-8")
)
