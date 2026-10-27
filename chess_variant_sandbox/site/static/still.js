// UP2 and UP4, `docs/plan-user-page.md` §5: a position on a server-drawn page, as the
// application's still board draws it (`stillboard.js` `boardPreview`, the one drawing of a
// board; nothing here works out a square). Loaded after `pages.BOARD_SCRIPTS` and before the
// page's own script, by the user page and by a member's page.
//
// The three globals the application's `core.js` gives its still board, which these pages
// load without it: no board here is turned round, the drawings arrive with the payload
// (`pieceArt`, as the listing sends them), and every name a board writes goes through `esc`,
// as it does on the application's pages.
let FLIPPED = false;
let PIECE_ART = {};
function esc(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// `still` is a payload the server drew (a game's opening, or a running game's position
// replayed through the engine) with each piece's drawing named by stem; `last` is the
// engine's last move, tinted as the play board tints it; `hand` keeps the hand strip under
// the board. A thumbnail (`plain`) draws no marks and names no square. The board's string is
// the only markup these pages put in: every name in it went through `esc`. Absent where
// there is no position, or no board script (a test's stand-in DOM).
// `moving` draws each piece in a group of its own (`preview.motion`, as the landing's boards
// are), so the user page's one motion (`profile.js` `slideOnSight`) has
// one element to move.
function stillNode(still, hash, last, hand, moving) {
  const box = document.createElement('div');
  box.className = 'stillBox';
  if (!still || typeof boardPreview !== 'function' || !document.createRange) return box;
  const icons = (still.icons || []).map(pair => (pair || []).map(icon =>
    icon && PIECE_ART[icon.stem]
      ? { href: PIECE_ART[icon.stem], rotate: icon.rotate, stem: icon.stem } : null));
  const drawn = boardPreview(Object.assign({}, still, {
    icons, hash, plain: true, last: last || null, handsUsed: !!hand && !!still.handsUsed,
    motion: !!moving,
  }), null);
  box.setAttribute('aria-hidden', 'true');
  box.appendChild(document.createRange().createContextualFragment(drawn));
  return box;
}

// UP4: the showcase's boards. The server wrote an empty `data-still` box where each game's
// board goes and handed the stills over as data (`pages._showcase_data`); each box takes its
// drawing here, once.
function fillStills(root, stills, art) {
  if (!root || !stills) return;
  Object.assign(PIECE_ART, art || {});
  for (const box of root.querySelectorAll('[data-still]')) {
    const hash = box.getAttribute('data-still');
    if (box.childElementCount || !stills[hash]) continue;
    const drawn = stillNode(stills[hash], hash, null, false);
    box.setAttribute('aria-hidden', 'true');
    while (drawn.firstChild) box.appendChild(drawn.firstChild);
  }
}
