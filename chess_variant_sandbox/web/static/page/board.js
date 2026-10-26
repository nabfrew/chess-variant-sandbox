// The board: drawing a position, animating a move, and playing one.

// ------------------------------------------------------------------ playing

// Where this board is drawn. The play page gives it the whole view; the designer gives
// it one panel of the column it edits in. Everything below this line is the same board
// either way -- the markup carries the same ids, and `document.getElementById` does not
// care which of the two wrote them.
//
// What the two do differ in is chrome. A page board is the only thing on screen, so it
// carries the crumb back to the game and the panel of rules and pieces beside it; an
// inline board sits beside the editor that already shows both, so it carries neither and
// gives that width to the board.
let BOARD_INLINE = false;

// Q993b: the third chrome, a try from a proven position drawn on the positions' own page
// (`proven.js`), with the list beside the board in place of the moves, rules and pieces.
// `BOARD_PROVEN` is that page's half (`how.proven`): its words, its *previous* and *next*,
// a new try drawn where this one was, and `moved`, which every snapshot of the try passes
// through so the list can tick its row from the finish's reply.
let BOARD_PROVEN = null;

// How long a button that ends the game (*resign*, *cancel the game*) stays armed after its
// first press, in ms. It was six seconds, and two of the first-visit study's readers met
// it reverting before their second press (N1). Fifteen gives a person time to read *really
// resign?* and decide. It is a timer at all because a phone's Safari gives a tapped
// button no focus, so *blur* never disarms it there: without one the button would stay
// armed until the next stray tap ended the game.
const ARM_HOLD = 15000;

// The board, given up. Everything it holds is about one game, so a view that takes the
// board away drops all of it: a reply still on its way lands on a session nobody is
// showing and is discarded by `maybeAgent`'s own check, and the next game opens without
// the last one's timeline behind it.
function leaveBoard() {
  if (BOARD_SIGHT) BOARD_SIGHT.disconnect();
  BOARD_SIGHT = null;
  BOARD_OFFSCREEN = false;
  SESSION = null;
  SHOWN = null;
  SELECTED = null;
  DROP = null;
  HIGHLIGHT = null;
  PHASE = null;
  NOTE = '';
  AT = null;
  CURSOR = null;
  PAUSED = false;
  TIMELINE = [];
  TIMELINE_FOR = null;
  BOARD_INLINE = false;
  BOARD_PROVEN = null;
  CORR_HELD = null;
  ROUND_SENDING = false;
  PROMO = null;
  PROMO_LOOKING = false;
  PROMO_PIECE = null;
  stopCorrPoll();
}

// `how` is `{ mount }` for a board drawn inside another view, `{ mount, proven }` for a try
// on the proven positions' own page (Q993b), and absent for the page's own board.
async function showBoard(snapshot, how) {
  stopPolling();
  stopCorrPoll();
  CORR_HELD = null;
  ROUND_SENDING = false;
  PROMO = null;
  PROMO_LOOKING = false;
  PROMO_PIECE = null;
  const proven = (how && how.mount && how.proven) || null;
  const inline = !!(how && how.mount) && !proven;
  BOARD_INLINE = inline;
  BOARD_PROVEN = proven;
  BOARD_FOCUS = null;
  const root = how && how.mount ? how.mount : view;
  SESSION = snapshot;
  SHOWN = null;
  SELECTED = null;
  DROP = null;
  HIGHLIGHT = null;
  PHASE = null;
  NOTE = '';
  // A first game opens on the rules (K3); `OPEN_TAB` is taken once.
  TAB = OPEN_TAB || 'moves';
  OPEN_TAB = null;
  const rules = snapshot.ruleset;
  // G3: the tab says which game is on the board. An inline board leaves its page's title.
  if (!inline && !proven && rules && rules.name) setTitle('Playing ' + rules.name);
  // A deep link to a game in progress never visited the picker, so the roster that
  // names the seats has to be fetched here as well. Without it a rule of thumb is
  // labelled with its own digest. A draft has no page to ask, so it takes the ladder
  // everything shares rather than a roster nobody can look up.
  await roster(rules.scratch ? '' : rules.hash).catch(() => {});
  // Your own side at the near end, which is the only orientation a person can read a
  // board in. A game with nobody in it, or with two people in it, keeps the first
  // player at the bottom; the flip button is there for both.
  FLIPPED = snapshot.seats[1] === 'human' && snapshot.seats[0] !== 'human';
  // A game between two people (Q570) is drawn from the reader's own side.
  const between = snapshot.correspondence || null;
  if (between) FLIPPED = between.you === 1;
  // A board opened starts in the frame its squares are named in. The shift is a way of
  // reading one board, not a preference carried between boards -- two games rarely have
  // the same width, and a remembered offset would land somewhere arbitrary.
  SHIFT = { files: 0, ranks: 0 };
  const mixed = snapshot.seats[0] !== snapshot.seats[1];
  // Nobody is at this board: both sides are the machine's and the game is there to be
  // read rather than played. It is what the watch control starts, and it is why that
  // control is offered on every other board and not on this one.
  const watched = !snapshot.seats.includes('human');
  // Whether this game counts is `snapshot.mode`, as the store says (`site/matches.py`),
  // and `frameHtml` reads it: a rated game has no take-back, a sandbox game has one, and
  // both can be resigned; the local application has no modes at all and sends nothing,
  // so it gets no mark and no resignation and keeps its take-back.
  //
  // The editor's trial says the *game* is not recorded, never that the rules are not
  // saved: they may well be, and after a save that read as a contradiction (Q576, N4).

  root.innerHTML = `
    ${inline ? `<div class="muted small">This is a game of the rules as they stand right
      now. Nothing about it is kept, and changing the draft ends it.</div>`
      : proven ? '' : `<button class="crumb" id="back">&larr; ${
        rules.scratch ? 'back to the editor'
          : snapshot.puzzle ? 'proven positions on ' + esc(rules.name) : esc(rules.name)}</button>`}
    ${!inline && rules.scratch ? `<div class="muted small u-mt-8">
      This is a trial of <b>${esc(rules.name)}</b>, and this trial isn't recorded: nothing
      about it is kept, and going back to the editor leaves no trace of it.</div>` : ''}
    ${snapshot.demo ? `<div class="ok u-mt-8">${esc(snapshot.demo.says)}
      The computer played these moves. Step forward to watch them, or move from here.</div>` : ''}
    ${snapshot.puzzle && !proven ? `<div class="ok u-mt-8" id="puzzleSays"
      aria-live="polite">${esc(puzzleSaid(snapshot))}</div>` : ''}
    <div class="split${inline || proven ? ' alone' : ''} u-mt-10">
      <div class="play-frame">
        ${inline ? `<div class="row small u-mb-8">
          <div>${seatHtml(0)}</div><span class="muted">vs</span><div>${seatHtml(1)}</div>
        </div>` : ''}
        ${frameHtml(inline, snapshot.mode, watched, between, (!!snapshot.round || !!snapshot.spy) && !watched, !!snapshot.puzzle)}
        <div class="row u-mt-8" id="review"></div>
        ${between ? '' : `<div class="row u-mt-8">
          ${proven ? `<button id="provenPrev" class="quiet" title="the position before this one on the list"
            >${esc(proven.words.previous)}</button><button id="provenNext" class="quiet"
            title="the position after this one on the list">${esc(proven.words.next)}</button>` : ''}
          ${snapshot.puzzle ? `<button id="again"
            title="a new practice game from the same proven position, against hardest"
            >try it again from the position</button>` : `<button id="again">new game</button>
          ${mixed ? `<button id="swap" title="the same game from the other side">
            play the other side</button>` : ''}
          ${watched ? '' : `<button id="watch" title="${esc(watchTitle())}">
            watch it play itself</button>`}`}
        </div>`}
        <div id="ground" class="muted small u-mt-6"></div>
        ${proven ? `<div class="ok small u-mt-8" id="puzzleSays"
          aria-live="polite">${esc(puzzleSaid(snapshot))}</div>` : ""}
      </div>
      ${inline || proven ? '' : `<div class="panel boardSide">
        <div class="seatLines">
          <div>${seatHtml(0)}</div>
          <div class="u-mt-6">${seatHtml(1)}</div>
          ${modeHtml(snapshot)}
        </div>
        <div class="tabs u-mt-10">
          ${['moves', 'rules', 'pieces'].map(tab => `<button data-tab="${tab}"${
            tab === TAB ? ' class="on"' : ''}>${tab}</button>`).join('')}
        </div>
        <div class="tabpane" id="pane"></div>
      </div>`}
    </div>`;

  // A draft is not in the library, so every one of these three goes through the editor's
  // own scratch route instead of through a hash nothing can resolve.
  //
  // `pending` fires here rather than in each button, so both of them -- new game and
  // play the other side -- show it exactly once, before the one slow call either way.
  //
  // This is the rematch control, so it names the game it was pressed on and the store
  // records the new match as that one's rematch (`matches.rematch_of`). Starting from a
  // game page or from the editor sends nothing, and those rows stay null. Only the match
  // id goes, because the server already knows who is asking and the match it names
  // already holds both seats.
  //
  // A game nobody sat at is nobody's to play again, and neither is a watch started from
  // one, so both leave the column null and the store refuses either as a rematch.
  //
  // The new game is played in the mode this one was. Sending nothing meant the server's
  // default, so a rematch of a rated game was a sandbox game and the board silently
  // stopped counting (Q249). A game nobody sits at is the exception and sends no mode:
  // nobody plays it, so there is nothing to rate, and the store refuses a rated watch.
  const restart = (seats, follows = true) => {
    // An inline board is always a draft's, and it restarts where it stands: the editor
    // puts the spinner in the board's own panel, because the page's own is at the top of
    // a form that is several screens tall.
    if (inline) return playDraftHere(seats);
    pending('Starting the game…');
    if (rules.scratch) return playDraft(seats);
    // A try from a proven position starts again from the position (R9b): the server seats
    // it, so the body names the game and the position and nothing else.
    if (snapshot.puzzle) {
      return api('/api/session', { hash: rules.hash, puzzle: snapshot.puzzle.id })
        .then(next => (proven ? proven.again(next) : open_(next))).catch(fail);
    }
    const body = { hash: rules.hash, seats };
    if (snapshot.mode && seats.includes('human')) body.mode = snapshot.mode;
    if (follows && !watched) body.rematch_of = snapshot.id;
    return api('/api/session', body).then(open_).catch(fail);
  };
  // Never by presence: the designer has a crumb of its own called `back`, and an inline
  // board that looked one up would rewire the editor's way out of the page.
  if (inline) document.getElementById('stopPlay').onclick = () => stopPlayingHere();
  else if (proven) {
    document.getElementById('provenPrev').onclick = () => proven.previous();
    document.getElementById('provenNext').onclick = () => proven.next();
  } else document.getElementById('back').onclick = () =>
    go(rules.scratch ? '#/design'
      : snapshot.puzzle ? '#/proven/' + rules.hash + '/' + snapshot.puzzle.id
        : '#/game/' + rules.hash);
  const again = document.getElementById('again');
  if (again) again.onclick = () => restart(snapshot.seats);
  // A rematch with the seats exchanged, which is the whole of "let me try that as the
  // other side" and is a new game rather than a redraw: the two armies are different
  // pieces and the position is the opening one either way.
  if (mixed && !between && !snapshot.puzzle) document.getElementById('swap').onclick = () =>
    restart([snapshot.seats[1], snapshot.seats[0]]);
  // Both sides to the same searching player, at the budget the designer's watch uses --
  // `WATCH_AGENT`, so the page holds one opinion about who is cheap enough to watch.
  const watch = document.getElementById('watch');
  if (watch) watch.onclick = () => restart([WATCH_AGENT, WATCH_AGENT], false);
  document.getElementById('flip').onclick = () => { FLIPPED = !FLIPPED; drawBoard(); };
  wireFullButton();
  // A refused take-back is answered under the button that asked for it. A rated game
  // draws no take-back at all -- the store refuses one, and a control whose only answer
  // is a refusal is worse than no control.
  const undo = document.getElementById('undo');
  if (undo) undo.onclick = () =>
    api('/api/session/' + snapshot.id + '/undo', {})
      .then(s => applySnapshot(s, false)).catch(error => fail(error, 'undo'));
  // Resigning is a finish: the row records it as a loss and, in a rated game, the rating
  // moves on it exactly as it moves on a checkmate. So it takes two presses, the first of which says
  // what the second does -- and the button disarms itself on a timer and when it loses
  // focus, because a control left armed behind the reader's back is one stray click away
  // from ending their game. The timer is `ARM_HOLD`.
  const resign = document.getElementById('resign');
  if (resign) {
    const disarm = () => { resign.dataset.armed = ''; sayOn(resign, 'resign'); };
    resign.onblur = disarm;
    resign.onclick = () => {
      if (resign.dataset.armed !== 'yes') {
        resign.dataset.armed = 'yes';
        sayOn(resign, 'really resign?');
        setTimeout(disarm, ARM_HOLD);
        return;
      }
      disarm();
      api('/api/session/' + snapshot.id + '/resign', {})
        .then(s => applySnapshot(s, false)).catch(error => fail(error, 'resign'));
    };
  }
  // A draw offered to the computer (Q790). An offer the rules do not allow yet is answered
  // here with the server's own sentence from the snapshot, costing no request; one that
  // goes is answered by the core, and a decline stays in the strip until the next move.
  const offerDraw = document.getElementById('draw');
  if (offerDraw && !between) offerDraw.onclick = () => {
    const offer = SESSION && SESSION.drawOffer;
    if (offer && !offer.may) { fail(new Error(offer.why), offerDraw); return; }
    offerDraw.disabled = true;
    api('/api/session/' + snapshot.id + '/draw', {})
      .then(s => {
        applySnapshot(s, false);
        const said = s.drawOffer && s.drawOffer.said;
        if (said && !s.outcome) { NOTE = said; drawStatus(); }
      })
      .catch(error => { drawStatus(); fail(error, offerDraw); });
  };
  // Scoped to whatever this board was drawn into, because an inline one has no tabs of
  // its own and the view around it may.
  root.querySelectorAll('[data-tab]').forEach(node => {
    node.onclick = () => {
      TAB = node.dataset.tab;
      root.querySelectorAll('[data-tab]').forEach(n => n.classList.toggle('on', n === node));
      drawPane();
    };
  });

  // A game you have not seen before starts running and starts unreviewed. Coming back
  // to one you were already watching -- the browser's Back button out of the rules and
  // in again -- keeps its timeline and keeps it paused if that is how you left it,
  // because otherwise looking anything up costs you your place.
  if (TIMELINE_FOR !== snapshot.id) PAUSED = false;
  CURSOR = null;
  AT = null;
  record(snapshot);
  wireReviewKeys();
  wireFit();

  // The rules are wanted *while playing*, so they are fetched now and not when a tab is
  // first opened: a panel that takes a round trip to appear is a panel nobody opens
  // twice. It is one request per ruleset per page load either way.
  detailFor(rules.hash).then(drawPane).catch(() => {});
  // A move picked between two people and never sent is still on the board when the
  // person comes back to it, in this browser, until it is sent or put back (Q733).
  if (between) CORR_HELD = corrHeldKept(snapshot);
  drawBoard();
  drawPane();
  watchBoardSight(root);
  maybeAgent();
  wireCorr();
  pollCorr();

  // Who is playing, on a board nobody sits at. A watch raises one question -- why are
  // the moves like that -- and the seat is the answer, so the sentence belongs to every
  // watch rather than to the designer's alone (Q265). It is the server's, off the
  // picker's own labels, and absent from every game a person is in.
  noteWatching(snapshot);

  // A demonstration opens on the position before the moves it shows, once.
  const demo = snapshot.demo;
  if (demo && !demo.opened) {
    demo.opened = true;
    const beat = TIMELINE.map(entry => entry.plies).lastIndexOf(demo.start);
    if (beat >= 0 && beat < TIMELINE.length - 1) showBeat(beat);
  }
}

// The play frame (plan-visual.md §3.3), one markup for both chromes: the game's state
// above the board, the board, and this game's own controls directly under it, so all
// three are in the first screen at a desk and on a phone (the first-visit study's B1,
// F1, F2). Every slot carries `data-frame`, which is what the look check reads a page by
// and what the play frame's restyle (V4) hangs its look on.
//
// Above the board, the status strip -- whose move, *in check*, and the result once the
// game is over -- and the legend line naming every mark the board carries, the
// promotion zone's toggle at its end. Over it, when a move promotes, the choice of what
// the piece becomes (`choosePromotion`, Q765), and when a strike places its survivor, the
// choice of where (`chooseLanding`, Q769). Under it, the hand (pieces held), then take
// back, flip, the view's shift where the board wraps, and resign last. Starting another game is not a control of this one, so new game, the
// other side and watch sit further down.
//
// `mode` is the store's word for the game (`snapshot.mode`). A sandbox game can be given
// up too (the first-visit study's F12): a lost game had take back and new game and no
// way to close it with a result. Its resignation is the same store call as a rated
// one's, and `site/rating.py::finish` leaves a sandbox row unrated, so it ends the game
// and moves nothing. The local application has no modes and no store to record a
// resignation in, so it offers none; a game nobody sits at is nobody's to give up.
//
// A game between two people (Q570, `between`) has no take-back, ever; its bar gains
// *offer a draw* (and *decline the draw* while the other's offer stands, Q790) and, until
// both have moved, *cancel the game*, with resign still last and still the one danger
// button (docs/archive/plan-return.md §1.8).
//
// A game against the computer gains *offer a draw* too, before resign (Q790): the
// computer answers at once, accepting only where the strongest search the game has reads
// the position as level (`web/draw_offer.py`, docs/reference/intransitive-quiet-endgame.md
// §e.1). When an offer may be made is the snapshot's (`drawOffer.may`, `drawOffer.why`), and
// the answer its sentence (`drawOffer.said`); the page judges nothing.
//
// V4 gave it its look, in plan-visual.md §3.3's order: the status strip, the board, the
// control bar, the legend line. The square names stand outside the squares (`drawCoords`),
// a finished game puts its result on a banner over the board (`drawEnded`), and on a
// phone the bar is a row of icons with the word under each (`FRAME_ICONS`).
function frameHtml(inline, mode, watched, between, rounds, puzzle) {
  const rated = mode === 'leaderboard';
  const ends = (rated || mode === 'sandbox') && !watched;
  return `
    <div class="row frame-status${between ? ' between' : ''}" data-frame="status"><span id="status"></span></div>
    <div class="board-row" id="boardRow"><div class="panel board-panel"><div class="board-frame"><div class="board-box">
      <svg id="board" data-frame="board" viewBox="0 0 100 100"></svg>
      <div class="coords" id="coords" aria-hidden="true"></div></div>
      <div class="frame-ended" id="ended" hidden></div>
      <div class="promo" id="promo" data-frame="promotion" role="dialog" aria-modal="true"
        aria-labelledby="promoHead" hidden></div></div></div>
    ${(between && between.you !== null) || rounds ? `<div class="corr-send" id="corrSend"
      data-frame="send" aria-live="polite"></div>` : ''}</div>
    <div id="hand" class="hand"></div>
    <div id="pieceSays" class="muted small piece-says" hidden></div>
    <div class="row frame-controls" data-frame="controls">
      ${inline ? `<button id="stopPlay" class="quiet" title="put the board back to editing this game"
        >${barWords('edit', 'back to editing')}</button>` : ''}
      ${rated || between || puzzle ? '' : `<button id="undo" class="quiet" title="take your last move back"
        >${barWords('undo', 'take back')}</button>`}
      <button id="flip" class="quiet" title="turn the board around">${barWords('flip', 'flip')}</button>
      ${inline ? '' : `<button id="fullBoard" class="quiet" title="${
        BOARD_FULL ? FULL_TITLE_OUT : FULL_TITLE_IN}">${fullWords(BOARD_FULL)}</button>`}
      <span class="row shift" id="shift"></span>
      ${between && between.you !== null ? `<button id="draw" class="quiet"
        title="offer a draw; it stands until the other player accepts it, declines it or moves"
        >${barWords('draw', 'offer a draw')}</button><button id="declineDraw" class="quiet" hidden
        title="say no to the other player's draw offer; the game goes on"
        >${barWords('decline', 'decline the draw')}</button>${between.mayCancel ? `<button id="cancel" class="quiet"
        title="call the game off; possible until both of you have moved, and it counts for nothing"
        >${barWords('cancel', 'cancel the game')}</button>` : ''}<button id="rematch" hidden
        title="challenge the other player to this game again, with the sides swapped"
        >${barWords('again', 'rematch')}</button>` : ''}
      ${ends && !between ? `<button id="draw" class="quiet"
        title="offer the computer a draw; it answers at once, by the strongest search this game has"
        >${barWords('draw', 'offer a draw')}</button>` : ''}
      ${ends && !(between && between.you === null) ? `<button id="resign" class="danger"
        title="${rated ? 'give the game up; it is recorded as a loss and your rating moves'
          : 'give the game up and end it here; a sandbox game counts for nothing'}"
        >${barWords('resign', 'resign')}</button>` : ''}
    </div>
    <div class="frame-legend small" data-frame="legend" id="legendRow" hidden>
      <span id="legend"></span><span id="marks"></span></div>`;
}

// The control bar's icons, drawn here rather than fetched: one 24-unit grid, a 1.8-unit
// round stroke in the button's own colour, no fill, so every icon is the same weight and
// takes the button's kind (quiet green, danger red) with it. Shown only on a phone, above
// the word, which stays (plan-visual.md §3.3: icons with a word under each, lower case).
const FRAME_ICONS = {
  undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
  flip: '<path d="M7 4v16"/><path d="m3 8 4-4 4 4"/><path d="M17 20V4"/><path d="m21 16-4 4-4-4"/>',
  resign: '<path d="M5 21V4"/><path d="M5 4h11l-2 4 2 4H5"/>',
  edit: '<path d="M15 5l4 4L9 19H5v-4z"/><path d="m13 7 4 4"/>',
  draw: '<path d="M5 12h14"/><circle cx="12" cy="7" r="1.4"/><circle cx="12" cy="17" r="1.4"/>',
  cancel: '<circle cx="12" cy="12" r="8"/><path d="m9 9 6 6M15 9l-6 6"/>',
  decline: '<path d="M5 12h14"/><path d="m7 19 10-14"/>',
  again: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 5v6h-6"/>',
  full: '<path d="M4 9V4h5"/><path d="M20 9V4h-5"/><path d="M4 15v5h5"/><path d="M20 15v5h-5"/>',
  unfull: '<path d="M9 4v5H4"/><path d="M15 4v5h5"/><path d="M9 20v-5H4"/><path d="M15 20v-5h5"/>',
};

function frameIcon(name) {
  return `<svg class="ico" viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"
    fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"
    stroke-linejoin="round">${FRAME_ICONS[name]}</svg>`;
}

// A bar button's face: its icon, then its word in a span of its own, so a button that
// changes what it says (resign asking to be pressed again) changes the word and keeps
// the icon.
function barWords(icon, words) {
  return `${frameIcon(icon)}<span class="w">${esc(words)}</span>`;
}

function sayOn(button, words) {
  const word = button.querySelector ? button.querySelector('.w') : null;
  if (word) word.textContent = words; else button.textContent = words;
}

// The board's side, fitted to the screen (S2): never wider than its column, and never so
// tall that the frame's controls under it fall off the first screen. The page's own
// board is measured from where it sits on the page, so the header and the status strip
// above it count; a board inline in the designer is wherever the editor put it, so only
// the frame's own parts count there. Never smaller than a phone's board, so a short
// landscape screen scrolls rather than shrinking the game to a stamp -- but for one line
// (`PIECE_SAYS_LINE`) where that line brings the whole frame onto the screen (Q809).
const BOARD_FLOOR = 280;
// The height of the one line a tapped piece's sentences take under the board (K3). It is
// kept for the line whether or not a piece is picked, and the line's own height is never
// read: a sentence that wraps to two or three lines once moved the board under the
// pointer, so a second tap could land on another square than the one aimed at (Q809).
const PIECE_SAYS_LINE = 22;
// The status strip's width under which its longest sentences (a note, the computer being
// slow, *in check*) take three lines rather than two.
const STRIP_NARROW = 560;
// Q761: the width the slot that sends a move between two people needs for its two buttons
// side by side, with its words over them. Under a board narrower than this (a tall board,
// height-limited) the slot and the strip take this much of the column rather than the
// board's width, where *Change my move* was squeezed to three clipped lines.
const SEND_WIDE = 440;
// ... and the least it takes beside the board, its buttons one over the other, where the
// column has the room: a tall board then keeps the height the slot took under it.
const SEND_BESIDE = 260;
const BESIDE_GAP = 16;

// A line of the status strip, as `.frame-status` sets it (`--strip-lines` lines of 22 px).
const STRIP_LINE = 22;

// Q761: a fit is worked out for the layout it ends in, not read off the one before. The
// strip's height is the one its width there gives it (a third or fourth line under
// `STRIP_NARROW`), and the frame's wood, a share of the panel's width, is read again until
// the width stops moving. Before, both came from the last layout, so a game opened with a
// board one size and the first move picked shrank it by a few pixels, and a frame could
// settle in either of two layouts depending on the one it came from.
function fitBoard() {
  const svg = document.getElementById('board');
  if (!svg || typeof svg.getBoundingClientRect !== 'function') return;
  const box = String(svg.getAttribute('viewBox') || '').split(/\s+/).map(Number);
  const panel = svg.closest ? svg.closest('.board-panel') : null;
  const controls = document.querySelector('[data-frame="controls"]');
  const hand = document.getElementById('hand');
  const status = document.querySelector('[data-frame="status"]');
  const legend = document.querySelector('[data-frame="legend"]');
  if (box.length !== 4 || !(box[2] > 0) || !(box[3] > 0) || !panel || !controls) return;
  const ratio = box[3] / box[2];
  // The board alone (Q638) is sized to the view's room, not the page's.
  if (BOARD_FULL && !BOARD_INLINE) { fitFull(svg, panel, ratio); return; }
  const tall = node => (node && node.getBoundingClientRect
    ? node.getBoundingClientRect().height : 0);
  const bottom = node => (node && !node.hidden && node.getBoundingClientRect
    ? node.getBoundingClientRect().bottom : 0);
  const row = document.getElementById('boardRow');
  const send = document.getElementById('corrSend');
  const sending = !!(send && !send.hidden && row && row.classList);
  const says = document.getElementById('pieceSays');
  for (let pass = 0; pass < 4; pass += 1) {
    // Measured with the slot under the board, where every layout starts.
    if (row && row.classList) row.classList.remove('beside');
    panel.style.width = '';
    // The panel's own padding and border, round the board, and the margin the square
    // names stand in.
    const rim = tall(panel) - tall(svg);
    // Read off the drawn boxes rather than the rounded `offsetWidth`, so a second fit of
    // an unchanged frame gives the same width (Q733: it lost a pixel each time).
    const wood = (panel.getBoundingClientRect().width - svg.getBoundingClientRect().width) || 0;
    const across = row && row.getBoundingClientRect ? row.getBoundingClientRect().width : 0;
    // Under the board: the slot that sends a move, the hand, the control bar and the
    // legend line, held to the first screen (S2). Read off where they end, so their
    // margins and a legend that wraps count too. A line is kept for a tapped piece's
    // sentences (K3, `drawPieceSays`), so picking a piece never pushes the bar off the
    // screen; where the sentences are shown, the room they take is read out again and the
    // one line kept instead, so the board is the same size whatever is picked (Q809).
    const panelBottom = panel.getBoundingClientRect().bottom;
    const last = Math.max(bottom(hand), bottom(controls), bottom(legend));
    const saysStyle = says && !says.hidden && typeof getComputedStyle === 'function'
      ? getComputedStyle(says) : null;
    const saysRoom = says && !says.hidden
      ? tall(says) + ((saysStyle && parseFloat(saysStyle.marginTop)) || 0) : 0;
    const under = (last > panelBottom ? last - panelBottom - saysRoom
      : tall(hand) + tall(controls)) + 12 + PIECE_SAYS_LINE;
    // Above it: the page's own board is measured from where it sits, so the header and
    // the strip count; a board inline in the designer counts only the frame's own strip.
    // The strip's height is the one it will have, a line more when it is narrow.
    const strip = tall(status);
    const wasNarrow = !!(status && status.classList && status.classList.contains('narrow'));
    const stripAt = narrow => (strip > 0 ? strip + ((narrow ? 1 : 0) - (wasNarrow ? 1 : 0)) * STRIP_LINE : 0);
    const top = BOARD_INLINE ? 16 : panel.getBoundingClientRect().top + (window.scrollY || 0) - strip;
    const sideAt = (narrow, less) => window.innerHeight - top - stripAt(narrow) - rim - less;
    // The strip above is the board's own, so it is no wider than the board: a verdict
    // boxed across the whole column read as a line of the page rather than of this game.
    // So is the slot under it that sends a move between two people (Q733). But never
    // narrower than the slot's two buttons need (Q761).
    const ownWidth = side => Math.max(side / ratio + wood, Math.min(across, SEND_WIDE));
    const narrowAt = side => strip > 0
      && (across > 0 ? Math.min(across, ownWidth(side)) : ownWidth(side)) < STRIP_NARROW;
    // Q809: a tall board on a phone (ten by eleven between two people, under the taller
    // strip and the slot that sends a move) stood at the floor with its legend's last row
    // 4 px under the first screen. Where the frame needs no more than the line kept for a
    // tapped piece below the floor, the board gives that much: it takes the height the
    // frame leaves it without that line, so the legend is on the first screen and only a
    // picked piece's sentences, while shown, push it down. A board further off the screen
    // than that keeps the floor and the page scrolls, as before.
    const floored = narrow => {
      const room = sideAt(narrow, under);
      if (room >= BOARD_FLOOR) return room;
      const tight = sideAt(narrow, under - PIECE_SAYS_LINE);
      return tight >= BOARD_FLOOR - PIECE_SAYS_LINE ? Math.min(BOARD_FLOOR, tight) : BOARD_FLOOR;
    };
    let side = floored(false);
    let narrow = narrowAt(side);
    if (narrow) side = floored(true);
    // Q761: a board held short by the screen's height (a tall one, a torus with its
    // copies) left most of a desk's column empty while the slot under it took 150 px of
    // that height. Beside the board the slot frees them, so it goes there when that gives
    // the bigger board: the board's height without the slot under it, or the width the
    // column has left once the slot has its least, whichever is smaller. The strip then
    // spans the board and the slot.
    let beside = false;
    if (sending && across > 0) {
      const style = typeof getComputedStyle === 'function' ? getComputedStyle(send) : null;
      const freed = tall(send) + ((style && parseFloat(style.marginTop)) || 0);
      const spans = strip > 0 && across < STRIP_NARROW;
      const nextTo = Math.min(sideAt(spans, under - freed),
        (across - BESIDE_GAP - SEND_BESIDE - wood) * ratio);
      const below = Math.min(side, (across - wood) * ratio);
      if (nextTo >= BOARD_FLOOR && nextTo > below + 1) {
        beside = true;
        side = nextTo;
        narrow = spans;
      }
    }
    // The width that gives the board that height, plus the panel round it.
    const width = Math.ceil(side / ratio + wood) + 'px';
    const own = beside ? '' : Math.ceil(ownWidth(side)) + 'px';
    const settled = pass > 0 && panel.style.maxWidth === width && narrow === wasNarrow
      && (!status || !status.style || status.style.maxWidth === own);
    panel.style.maxWidth = width;
    if (beside) {
      row.classList.add('beside');
      panel.style.width = width;
    }
    if (status && status.style) status.style.maxWidth = own;
    if (send && send.style) send.style.maxWidth = own;
    // The strip's height is fixed, so the board never moves when the status changes
    // (Q725), and sized here, once a size: a strip narrower than `STRIP_NARROW` keeps a
    // third line, which is what its longest states take on a phone.
    if (status && status.classList) status.classList.toggle('narrow', narrow);
    if (settled) break;
  }
}

let FIT_WIRED = false;
function wireFit() {
  if (FIT_WIRED || typeof window.addEventListener !== 'function') return;
  FIT_WIRED = true;
  const refit = () => {
    if (SESSION) { fitBoard(); fitPortals(document.getElementById('board')); }
  };
  window.addEventListener('resize', refit);
  // Turning a phone round (Q638): Safari has been known to report the old size to the
  // `resize` that comes with it, so the fit is taken again once the turn has settled.
  window.addEventListener('orientationchange', () => setTimeout(refit, 250));
}

// ------------------------------------------------------------------ the board alone (Q638)

// The user, 2026-10-02, on their own iPhone (first-visit-study.md §1.2a): *"it would maybe be
// nice if it would put the board and only the board in full screen mode."* The bar's *full
// screen* turns the play frame into a view of its own over the page (`html.board-full`, the
// stylesheet's): the strip (whose move, the result), the board with its banner and its
// cards, the slot a round or a game between two people sends its move from (and where a
// round's order is said), the hand of a game with drops, the way out, and once the game is
// over the review row the banner's *review* steps with. Nothing about the
// game changes, so a move is played there as anywhere, and turning the phone round refits
// the board (`fitFull`) without leaving the view.
//
// The browser's Fullscreen API hides its bars where it has one. Where it has none (an
// iPhone's Safari) or refuses, the view is the same, filling the window: the view is the
// page's class either way, so the two can never disagree about what is drawn. A history
// entry is pushed on the way in, so a phone's back gesture leaves the view rather than the
// game; leaving by the control or by the browser's own way out takes it back off. Any
// other page leaves the view (`render`).
let BOARD_FULL = false;
let FULL_PUSHED = false;   // the history entry the view pushed, still on top
let FULL_NATIVE = false;   // the browser's own full screen is the view's
const FULL_TITLE_IN = 'the board alone, filling the screen; the moves are played as ever';
const FULL_TITLE_OUT = 'back to the page round the board';

function fullWords(on) {
  return on ? barWords('unfull', 'leave full screen') : barWords('full', 'full screen');
}

function fullscreenNow() {
  return document.fullscreenElement || document.webkitFullscreenElement || null;
}

// Settle what a promise-returning browser call answers, so a refusal is nobody's error.
function quietly(call) {
  try {
    const answer = call();
    if (answer && typeof answer.catch === 'function') answer.catch(() => {});
  } catch (error) { /* the view stands without it */ }
}

// `how` says who asked: 'button', 'fullscreen' (the browser left its own), 'popstate' (the
// back gesture took the entry off already) or 'route' (another page is being drawn).
function setBoardFull(on, how) {
  on = !!on;
  if (on === BOARD_FULL) return;
  BOARD_FULL = on;
  const root = document.documentElement;
  if (root && root.classList) root.classList.toggle('board-full', on);
  const button = document.getElementById('fullBoard');
  if (button) {
    button.innerHTML = fullWords(on);
    button.title = on ? FULL_TITLE_OUT : FULL_TITLE_IN;
  }
  if (on) {
    wireFull();
    try {
      history.pushState({ boardFull: true }, '');
      FULL_PUSHED = true;
    } catch (error) { FULL_PUSHED = false; }
    const ask = root && (root.requestFullscreen || root.webkitRequestFullscreen);
    if (ask && (document.fullscreenEnabled || document.webkitFullscreenEnabled)) {
      quietly(() => ask.call(root, { navigationUI: 'hide' }));
    }
  } else {
    if (how !== 'fullscreen' && fullscreenNow()) {
      const out = document.exitFullscreen || document.webkitExitFullscreen;
      if (out) quietly(() => out.call(document));
    }
    FULL_NATIVE = false;
    const pushed = FULL_PUSHED;
    FULL_PUSHED = false;
    if (pushed && (how === 'button' || how === 'fullscreen')) {
      try { history.back(); } catch (error) { /* the entry stays; it is this page's */ }
    }
  }
  if (SESSION) { fitBoard(); fitPortals(document.getElementById('board')); }
}

function wireFull() {
  if (wireFull.wired || typeof document.addEventListener !== 'function') return;
  wireFull.wired = true;
  const changed = () => {
    if (fullscreenNow()) { if (BOARD_FULL) FULL_NATIVE = true; return; }
    if (BOARD_FULL && FULL_NATIVE) setBoardFull(false, 'fullscreen');
  };
  document.addEventListener('fullscreenchange', changed);
  document.addEventListener('webkitfullscreenchange', changed);
  if (typeof window.addEventListener === 'function') {
    window.addEventListener('popstate', event => {
      if (!BOARD_FULL || (event && event.state && event.state.boardFull)) return;
      FULL_PUSHED = false;
      setBoardFull(false, 'popstate');
    });
  }
}

function wireFullButton() {
  const button = document.getElementById('fullBoard');
  if (button) button.onclick = () => setBoardFull(!BOARD_FULL, 'button');
}

// The board as large as the view leaves it. Upright, the strip and the way out share the
// top row and the slot and the hand stand under the board; on its side the board takes the
// height and those stand in a column beside it. Which one is the window's shape, set here
// as `.wide` so the stylesheet and this arithmetic agree. The panel's rim is read off the
// drawn boxes, and its wood, a share of its width, again until the width stops moving.
function fitFull(svg, panel, ratio) {
  const frame = panel.closest ? panel.closest('.play-frame') : null;
  if (!frame || typeof frame.getBoundingClientRect !== 'function') return;
  const status = document.querySelector('[data-frame="status"]');
  const controls = document.querySelector('[data-frame="controls"]');
  const send = document.getElementById('corrSend');
  const hand = document.getElementById('hand');
  const review = document.getElementById('review');
  const row = document.getElementById('boardRow');
  const wide = window.innerWidth > window.innerHeight;
  if (frame.classList) frame.classList.toggle('wide', wide);
  if (row && row.classList) row.classList.remove('beside');
  if (status && status.style) status.style.maxWidth = '';
  if (send && send.style) send.style.maxWidth = '';
  const styleOf = node => (typeof getComputedStyle === 'function' ? getComputedStyle(node) : {});
  const px = value => parseFloat(value) || 0;
  const shown = node => !!(node && !node.hidden && typeof node.getBoundingClientRect === 'function');
  const tall = node => (shown(node) ? node.getBoundingClientRect().height : 0);
  const outer = node => { const height = tall(node); return height > 0 ? height + px(styleOf(node).marginTop) : 0; };
  // The strip's lines follow its width, as they do above the page's own board.
  const across = shown(status) ? status.getBoundingClientRect().width : 0;
  if (status && status.classList) status.classList.toggle('narrow', across > 0 && across < STRIP_NARROW);
  const pad = styleOf(frame);
  const box = frame.getBoundingClientRect();
  const innerW = box.width - px(pad.paddingLeft) - px(pad.paddingRight);
  const innerH = box.height - px(pad.paddingTop) - px(pad.paddingBottom);
  const roomW = wide ? innerW - across - px(pad.columnGap) : innerW;
  const roomH = (wide ? innerH : innerH - Math.max(tall(status), tall(controls))
    - outer(send) - outer(hand) - outer(review))
    - px(styleOf(panel).marginTop);
  let width = Math.max(0, Math.floor(roomW));
  for (let pass = 0; pass < 4; pass += 1) {
    panel.style.width = width + 'px';
    panel.style.maxWidth = width + 'px';
    const outside = panel.getBoundingClientRect();
    const inside = svg.getBoundingClientRect();
    const wood = outside.width - inside.width;
    const rim = outside.height - inside.height;
    const next = Math.max(0, Math.floor(Math.min(roomW - wood, (roomH - rim) / ratio) + wood));
    if (next === width) break;
    width = next;
  }
  panel.style.width = width + 'px';
  panel.style.maxWidth = width + 'px';
}

function applySnapshot(snapshot, animate = true) {
  const previous = SHOWN;
  keepDepths(SESSION, snapshot);
  // An answer that moved nothing -- the computer sealing its move in a round (schema 26),
  // which leaves the board as it was -- draws the board as it stands rather than playing
  // the last move again.
  const moved = !SESSION || SESSION.id !== snapshot.id
    || (SESSION.history || []).length !== (snapshot.history || []).length;
  SESSION = snapshot;
  SELECTED = null;
  DROP = null;
  NOTE = '';
  // A piece tapped to see what it sees (schema 30) belongs to the position it was tapped on.
  if (moved) SIGHT_TAP = null;
  // A take-back is not a ply, so it leaves no trail: the marks that say where a wild
  // piece walked belong to a move that was made, and the position being restored is one
  // from before it.
  if (!animate) PHASE = null;
  // The game is recorded whether or not anyone is watching it happen. A reply that
  // lands while somebody is reading move six is still a move that was played, and
  // `record` reads the position it was played in off the timeline rather than off the
  // board, so a reader's place in the game cannot change what gets written down.
  record(snapshot);
  notePuzzle(snapshot);
  // A take-back rewrites the game being reviewed, so it ends the review.
  if (!animate && AT) { CURSOR = null; AT = null; }
  // Reviewing means the board is showing a position the game has already left, and a
  // new ply must not drag it forward. Everything that grew still grows: the move list,
  // and the review row's count of how far there now is to go.
  if (AT) { drawPane(); drawReview(); }
  else { drawBoard(animate && moved ? previous : null); drawPane(); }
  maybeAgent();
}

// A try from a proven position (R9b, `site/proven.py`): the server's sentence while it
// runs, and at its end the server's sentence for how it ended -- `won` where the engine's
// own outcome names the person's side the winner, `lost` at any other ending. The page
// reads the outcome the board already draws and decides nothing.
function puzzleSaid(snapshot) {
  const puzzle = snapshot.puzzle;
  if (!puzzle) return '';
  const ended = snapshot.outcome;
  if (!ended) return puzzle.said;
  return ended.winner === puzzle.side ? puzzle.won : puzzle.lost;
}

function notePuzzle(snapshot) {
  const line = document.getElementById('puzzleSays');
  if (line && snapshot.puzzle) line.textContent = puzzleSaid(snapshot);
  if (snapshot.puzzle && BOARD_PROVEN && BOARD_PROVEN.moved) BOARD_PROVEN.moved(snapshot);
}

// An agent that fails has to say so. This used to `.catch(fail)`, which prepends an error
// box and leaves everything else exactly as it was -- so the status still said "thinking",
// with a spinner, forever, and `onSquare` refuses human input while `awaitingAgent` is
// set. The game was unrecoverable and the screen said it was working. Reported from a
// real session against a stale build of the extension, where every agent move raised.
//
// **A reply belongs to the game it was asked of.** `new game` and `play the other side`
// both start a second session while a reply to the first is still on its way, and the
// board itself is gone the moment the page moves on. Neither used to be checked, so an
// abandoned game went on playing into whatever had replaced it: two machine games taking
// alternate plies on one board, pieces from the position twenty moves back appearing
// among the pieces of the new one, and captures with nothing on the board to explain
// them. Watched here as `first@53, second@1, first@54, second@2`, which is precisely
// what it looks like -- and the reason it never stopped is that each chain paints and
// then asks for the next move, so replacing the game starts a second chain rather than
// ending the first.
//
// **Paused means paused, and this is the only place it can be.** A machine game is that
// chain and nothing else: there is no timer and no loop to stop, so the game runs for
// exactly as long as something keeps asking for the next move. A reply already on its
// way still lands and is still written into the timeline -- it is a move that was
// played -- and stops there, because this is what would have asked for the one after it.
function maybeAgent() {
  if (PAUSED) return;
  if (!SESSION || !SESSION.awaitingAgent) return;
  // Nobody is looking: the tab is in the background or the board is scrolled out of
  // sight. The chain waits, and `boardSeen` starts it again through here (Q541: a watch
  // in the designer's bench asked 14 times in 8 s with its board off screen).
  if (boardUnseen()) return;
  const status = document.getElementById('status');
  if (!status) return;                        // the board is not on screen any more
  const id = SESSION.id;
  // One question at a time per game: the board coming back into sight while an answer
  // is on its way must not ask a second one beside it.
  if (AGENT_ASKING === id) return;
  AGENT_ASKING = id;
  AGENT_SLOW = null;
  const answered = () => {
    if (AGENT_ASKING === id) AGENT_ASKING = null;
    if (AGENT_SLOW === id) AGENT_SLOW = null;
  };
  const mine = () => !!SESSION && SESSION.id === id && !!document.getElementById('status');
  status.innerHTML = thinkingHtml() + (SESSION.inCheck ? checkPill() : '');
  // On the site, an answer slower than `SLOW_AFTER` is the box sharing its cores with
  // other games (`docs/archive/plan-launch.md` §5.2, L11): the strip says so once, and that the
  // move made is kept. No count, no queue place, nothing that grows.
  if (ACCOUNTS) setTimeout(() => {
    if (AGENT_ASKING !== id || !mine()) return;
    AGENT_SLOW = id;
    drawStatus();
  }, SLOW_AFTER);
  api('/api/session/' + id + '/agent', {})
    .then(s => whenSettled(() => { answered(); if (mine()) applySnapshot(s); })).catch(error => {
    answered();
    if (!mine()) return;
    // The sentence is a toast and the way out of it goes into the status. Never a box
    // beside the status: that grew the strip and pushed the board down (Q725).
    fail(error);
    status.innerHTML = '';
    const again = document.createElement('button');
    again.textContent = 'try that move again';
    again.onclick = () => { clearToast(); maybeAgent(); };
    // The seat's name, never its agent string (P15).
    status.append(`${seatName(SESSION.seats[SESSION.sideToMove])} could not move. `, again);
  });
}

// Whether the board on screen can be seen. Two things hide it, and either stops the
// machine chain: the tab going into the background, and the board leaving the viewport,
// which is what a watch in the designer's bench does when the person scrolls to the
// cards. Not `PAUSED`, which is the person's own pause and what the pause button shows;
// this one lifts by itself when the board is back.
let AGENT_ASKING = null;   // the session an agent request is on its way for
let AGENT_SLOW = null;     // that session, once its answer has taken `SLOW_AFTER`
const SLOW_AFTER = 3000;   // ms; the plan's three seconds
const SLOW_SAID = 'The computer is busy with other games; your move is safe.';
let BOARD_OFFSCREEN = false;
let BOARD_SIGHT = null;    // the IntersectionObserver on the board being drawn

function boardUnseen() {
  return BOARD_OFFSCREEN || document.visibilityState === 'hidden';
}

// Watches the board `root` holds, in place of any board watched before. Wired from
// `showBoard`; the tab's visibility is one listener for the page.
function watchBoardSight(root) {
  if (BOARD_SIGHT) BOARD_SIGHT.disconnect();
  BOARD_SIGHT = null;
  BOARD_OFFSCREEN = false;
  if (!watchBoardSight.wired && typeof document.addEventListener === 'function') {
    watchBoardSight.wired = true;
    document.addEventListener('visibilitychange', boardSeen);
  }
  const board = root && root.querySelector ? root.querySelector('#board') : null;
  if (!board || typeof IntersectionObserver !== 'function') return;
  const watching = new IntersectionObserver(entries => {
    if (BOARD_SIGHT !== watching || !entries.length) return;
    BOARD_OFFSCREEN = !entries[entries.length - 1].isIntersecting;
    boardSeen();
  });
  BOARD_SIGHT = watching;
  watching.observe(board);
}

// Back in sight: on with the game, through the one place that asks.
function boardSeen() {
  if (!boardUnseen()) maybeAgent();
}

// A snapshot nobody asked to see waits for the board to finish showing the previous
// move. There is exactly one of those -- the agent's reply -- and it is the whole of this
// bug: an agent answers in 40-550ms, a move takes 630ms to travel, and a capture's strike
// does not *start* until the capturing piece is 80% of the way there. So every capture a
// person made was wiped before anything of it had been drawn, while every capture the
// agent made played out in full, because nothing redraws the board on a person's turn.
// Reported exactly as it looks from the other side of the screen: "it only works when
// black captures".
//
// The wait is on the paint and not on the request, so the agent thinks *while* the board
// finishes moving and the delay costs nothing it did not already cost.
function whenSettled(paint) {
  setTimeout(paint, Math.max(0, BUSY_UNTIL - performance.now()));
}

function pieceOf(value) {
  if (!value) return null;
  const index = value - 1;
  // By `sideStride`, a whole army's values: with a facing, `index / nTypes` is the
  // frame a piece reads its moves from and not the player who owns it.
  const stride = SESSION.ruleset.sideStride || SESSION.ruleset.nTypes;
  return {
    player: Math.floor(index / stride),
    type: index % SESSION.ruleset.nTypes,
    facing: Math.floor((index % stride) / SESSION.ruleset.nTypes),
  };
}

// The position on screen, which is the beat under the review cursor when there is one
// and the game's own position otherwise. Every part of the drawing goes through these
// two, because a board taken from one position with a territory count taken from
// another would be two answers to one question.
function position() { return AT ? AT.board : SESSION.board; }
function tint() { return (AT ? AT.colour : SESSION.colour) || []; }
// Schema 22: where the folds stand, `{square, type, pair}` each (`fold_payload`); a
// reviewed beat that does not say keeps the game's own.
function foldsShown() { return (AT && AT.folds) || SESSION.folds || []; }

function pieceAt(square) { return pieceOf(position()[square]); }

// Schema 19: the compiled facts about a portal end's type (`portal_payload`), or null.
function portalEnd(type) {
  const portals = SESSION && SESSION.ruleset && SESSION.ruleset.portals;
  if (!portals) return null;
  return portals.ends.find(end => end.type === type) || null;
}

// The portal end standing on a square of the position on screen, with its partner and its
// words, or null. Where an end stands is the board's, as every piece's square is.
function portalOn(square) {
  if (square === null || square === undefined || !SESSION) return null;
  const piece = pieceAt(square);
  const end = piece ? portalEnd(piece.type) : null;
  if (!end) return null;
  const portals = SESSION.ruleset.portals;
  const partner = portalPartner(portals, end);
  const W = SESSION.ruleset.width;
  const name = at => squareName(at % W, Math.floor(at / W));
  const there = partner ? portalSquare(partner.type) : -1;
  const from = portalSources(portals, end).map(source => portalSquare(source.type))
    .filter(at => at >= 0).map(name);
  const words = portalWords(end, partner, name(square), portalPairs(portals), sideName,
                            there >= 0 ? name(there) : null, from,
                            portalNamer(SESSION.ruleset.pieceIds, SESSION.ruleset.pieceNames));
  return { end, partner, words };
}

// Schema 21 (`docs/archive/plan-portals-2.md` §5.2): whether the piece picked on the board may not
// go in by this end, so the end is dimmed and the doors it may use are the bright ones.
// Only in a game where some end lets only some pieces in: elsewhere the arms' colours say
// whose pieces go in, and a pair game looks as it always has. Read off the payload's
// `admits` (the compiled filter), never worked out.
function portalRefuses(end) {
  const portals = SESSION && SESSION.ruleset && SESSION.ruleset.portals;
  if (!portals || !end || SELECTED === null || AT) return false;
  if (!portals.ends.some(one => one.filtered)) return false;
  const piece = pieceAt(SELECTED);
  if (!piece || (piece.player !== 0 && piece.player !== 1) || portalEnd(piece.type)) return false;
  const id = (SESSION.ruleset.pieceIds || [])[piece.type];
  return !(((end.admits || [])[piece.player]) || []).includes(id);
}

// Schema 19: what the reader is looking at among the portals. `INSPECT` is the end last
// tapped, whose words stand under the board and whose partner is linked to it; `HOVER_END`
// the end under the pointer, linked while the reader has links on (`LINKS`, theirs, kept
// by the browser, off by default in play, §4); `LIT` the move whose exit the chooser
// lights (D6).
let INSPECT = null;
let HOVER_END = null;
let LIT = null;

// Schema 30 (Q999c, `docs/archive/plan-passive.md` §4.6): lines of sight. `SIGHT_HOVER` is the piece
// under the pointer whose sight is drawn; `SIGHT_TAP` the one tapped, `{square, ply}`, which
// holds until another tap or the next position. Both are of the live position only: the
// squares a piece sees are the core's (`sights`, `Engine.sight_of`), sent for the position in
// play, and a step back draws the effects' badges (`effects`) but no sight.
let SIGHT_HOVER = null;
let SIGHT_TAP = null;

// What the piece on `square` sees in the position in play, `[[square, bits], ...]`, or null:
// the snapshot's `sights`, the core's answer. Nothing while a step back is on screen.
function sightOf(square) {
  if (!SESSION || AT || square === null || square === undefined) return null;
  const found = (SESSION.sights || []).find(entry => entry.square === square);
  return found && found.seen.length ? found.seen : null;
}

// The tapped piece's square while it is of the position in play, else null.
function sightTapped() {
  return SIGHT_TAP && SESSION && SIGHT_TAP.ply === SESSION.ply && !AT ? SIGHT_TAP.square : null;
}

// The effects on the position on screen, `[[square, bits], ...]` for each piece under one:
// the live snapshot's `effects` (a value per square, Q999a), or the step back's (pairs, the
// history entry's or the replayed position's). Empty in a game without a sight.
function effectsShown() {
  return effectPairs(AT ? AT.effects : SESSION.effects);
}

function effectPairs(effects) {
  if (!effects || !effects.length) return [];
  if (Array.isArray(effects[0])) return effects.filter(pair => pair[1]);
  return effects.map((bits, square) => [square, bits]).filter(pair => pair[1]);
}

// Each affected piece's badges (`effectBadges`), on its square's corners.
function effectsHtml(affected, cell) {
  return affected.map(([square, bits]) => {
    if (cellAbsent(square, FRAME)) return '';
    const at = squareXY(square, cell, FRAME);
    const piece = pieceAt(square);
    return effectBadges(bits, at[0], at[1], cell, piece ? piece.player : undefined);
  }).join('');
}

// The legend's kinds for the effects on screen, and (schema 30) the sight's tint in every
// game that has one.
function effectLegendKinds(affected) {
  const kinds = new Set();
  for (const [, bits] of affected) for (const kind of effectKinds(bits)) kinds.add(kind);
  if (SESSION && SESSION.sights) kinds.add('seen');
  return [...kinds];
}

// What the piece under the pointer, else the one tapped, sees: each square tinted by the
// effect it would put on a piece there, and the piece's own square ringed. Laid over the
// drawing, as the pair links are, so a hover never redraws the board.
function paintSight() {
  const svg = document.getElementById('board');
  if (!svg || !SESSION || !FRAME || typeof svg.querySelector !== 'function') return;
  const old = svg.querySelector('[data-sightlayer]');
  if (old) old.remove();
  const square = SIGHT_HOVER !== null ? SIGHT_HOVER : sightTapped();
  const seen = sightOf(square);
  if (!seen) return;
  const group = document.createElementNS(SVGNS, 'g');
  group.setAttribute('data-sightlayer', String(square));
  group.setAttribute('pointer-events', 'none');
  group.innerHTML = sightHtml(square, seen, 100 / Math.max(SESSION.ruleset.width,
                                                           SESSION.ruleset.height));
  svg.appendChild(group);
}

// The drawing `paintSight` lays: every seen square tinted (`seenSquare`), then the seeing
// piece's square ringed in the colour of what it does.
function sightHtml(square, seen, cell) {
  let out = '';
  let union = 0;
  for (const [target, bits] of seen) {
    union |= bits;
    if (cellAbsent(target, FRAME)) continue;
    const at = squareXY(target, cell, FRAME);
    out += seenSquare(at[0], at[1], cell, bits, FRAME, false);
  }
  const at = squareXY(square, cell, FRAME);
  return out + seenSquare(at[0], at[1], cell, union, FRAME, true);
}
const LINKS_KEY = 'chess_variant_sandbox.portalLinks';
let LINKS = linksShown();

function linksShown() {
  try { return window.localStorage.getItem(LINKS_KEY) === '1'; } catch (err) { return false; }
}

function toggleLinks() {
  LINKS = !LINKS;
  try { window.localStorage.setItem(LINKS_KEY, LINKS ? '1' : '0'); } catch (err) { /* not offered */ }
  drawBoard();
}

function linkToggleHtml(shown) {
  const lit = shown ? ' class="u-bc-act u-c-act"' : '';
  return `<button id="linktoggle" aria-pressed="${!!shown}"${lit}
    title="an arrow from each portal to the one a piece going in there comes out beside">portal map</button>`;
}

// Where an end of this type stands in the position on screen, or -1: a momentum end on the
// board, a fold (schema 22) where the position's `folds` say.
function portalSquare(type) {
  if (isFold(portalEnd(type))) return foldSquare(type);
  const board = position();
  for (let square = 0; square < board.length; square++) {
    const piece = pieceOf(board[square]);
    if (piece && piece.type === type && portalEnd(piece.type)) return square;
  }
  return -1;
}

// Schema 22 (Q843d): where a fold end of this type stands in the position on screen, or -1.
function foldSquare(type) {
  const fold = foldsShown().find(one => one.type === type);
  return fold ? fold.square : -1;
}

// The fold lying on a square of the position on screen, with the end it leads to (`partner`,
// standing on `there`, -1 for none) and its words, or null. A fold is no board byte, so it
// is found in the position's `folds` (`fold_payload`), never in the board.
function foldOn(square) {
  if (square === null || square === undefined || !SESSION) return null;
  const fold = foldsShown().find(one => one.square === square);
  const end = fold ? portalEnd(fold.type) : null;
  if (!end) return null;
  const portals = SESSION.ruleset.portals;
  const partner = portalPartner(portals, end);
  const W = SESSION.ruleset.width;
  const name = at => squareName(at % W, Math.floor(at / W));
  const there = partner ? foldSquare(partner.type) : -1;
  const from = portalSources(portals, end).map(source => foldSquare(source.type))
    .filter(at => at >= 0).map(name);
  const words = foldWords(end, name(square), portalPairs(portals), sideName,
                          there >= 0 ? name(there) : null, from,
                          portalNamer(SESSION.ruleset.pieceIds, SESSION.ruleset.pieceNames));
  return { end, partner, there, words };
}

function boardCentre(square, cell) {
  const xy = squareXY(square, cell, FRAME);
  return [xy[0] + cell / 2, xy[1] + cell / 2];
}

// A line from (ax, ay) to (bx, by), each end held back by `trimA` and `trimB` cells, so
// it runs between two glyphs rather than over them; '' when nothing is left of it.
function trimmedLine(ax, ay, bx, by, trimA, trimB, cell, attrs) {
  const length = Math.hypot(bx - ax, by - ay);
  if (!(length > (trimA + trimB) * cell)) return '';
  const ux = (bx - ax) / length, uy = (by - ay) / length;
  return `<line x1="${(ax + ux * trimA * cell).toFixed(3)}" y1="${(ay + uy * trimA * cell).toFixed(3)}"
    x2="${(bx - ux * trimB * cell).toFixed(3)}" y2="${(by - uy * trimB * cell).toFixed(3)}" ${attrs}/>`;
}

// A move through a portal in its two legs (§4): the piece's own way to the end it went in
// by, and the king's step out from the other end to where it stands. Each leg is drawn
// along the step the mover took (`wrapStep`, its own seams), ending at the square, so a
// leg through a seam runs in from the far side as the piece did; and the way in is the way
// the engine says the piece went (`exitDir`, `portalWayIn`, as the trip slides it), which
// round a seam is not always the shortest (Q839). In the pair's colour (`--pair-N`, Q838b:
// which pair it went through, and never the last move's yellow-amber, which pair 2's hue
// is close to), cased in the glyph's edge so it reads on both woods.
function portalLegsHtml(last, cell, W, H) {
  if (!last || last.portal === undefined || last.portalOut === undefined
      || last.from === null || last.from === undefined) return '';
  const allow = wrapAllow(SESSION.ruleset.wrap, last.player);
  const way = portalOn(last.portal) || portalOn(last.portalOut);
  const ink = way ? `var(--pair-${way.end.pair})` : 'var(--hl-last)';
  const leg = (step, to, trimFrom, trimTo) => {
    const [bx, by] = boardCentre(to, cell);
    const [dx, dy] = stepXY(step, cell, FRAME);
    const ax = bx - dx, ay = by - dy;
    const cased = trimmedLine(ax, ay, bx, by, trimFrom, trimTo, cell,
      `stroke="var(--portal-edge)" stroke-width="${cell * 0.12}" stroke-linecap="round" opacity="0.55"`);
    if (!cased) return '';
    return cased + trimmedLine(ax, ay, bx, by, trimFrom, trimTo, cell,
      `stroke="${ink}" stroke-width="${cell * 0.065}" stroke-linecap="round"`);
  };
  const into = portalWayIn(wrapStep(last.from, last.portal, W, H, allow), last.exitDir, W, H);
  return `<g data-legs="${last.portal}" pointer-events="none">${
    leg(into, last.portal, 0.1, 0.45)}${
    leg(wrapStep(last.portalOut, last.to, W, H, allow), last.to, 0.45, 0.3)}</g>`;
}

// A move through a fold at rest (schema 24, `docs/archive/plan-portals-2.md` §5.3R): one leg in the
// group's colour, cased in the edge, from where it stood to the fold it went in by, and the
// ripple's ring on that fold. The move ended on the end the fold leads to, which takes the
// ordinary last-move mark as any landing does. The squares are the engine's (`foldIn`,
// `fold_transit`).
function foldLegsHtml(last, cell, W, H) {
  if (!last || last.foldIn === undefined || last.foldOut === undefined
      || last.from === null || last.from === undefined) return '';
  const allow = wrapAllow(SESSION.ruleset.wrap, last.player);
  const ink = `var(--pair-${last.foldPair || 1})`;
  const cased = `stroke="var(--portal-edge)" stroke-linecap="round" opacity="0.55"`;
  const [bx, by] = boardCentre(last.foldIn, cell);
  const [dx, dy] = stepXY(wrapStep(last.from, last.foldIn, W, H, allow), cell, FRAME);
  const line = attrs => trimmedLine(bx - dx, by - dy, bx, by, 0.1, 0.42, cell, attrs);
  const under = line(`${cased} stroke-width="${cell * 0.12}"`);
  const leg = under ? under + line(`stroke="${ink}" stroke-width="${cell * 0.065}" stroke-linecap="round"`) : '';
  const ring = `<circle cx="${bx}" cy="${by}" r="${cell * 0.4}" fill="none" ${cased} stroke-width="${cell * 0.11}"/>`
    + `<circle cx="${bx}" cy="${by}" r="${cell * 0.4}" fill="none" stroke="${ink}" stroke-width="${cell * 0.055}"/>`;
  return `<g data-foldlegs="${last.foldIn}" pointer-events="none">${leg}${ring}</g>`;
}

// The group map (schema 21, `docs/archive/plan-portals-2.md` §5.1; the pair link of §4 before it):
// an arrow from an end to the end it leads to, in the group's colour cased in the edge.
// With the map on (`LINKS`, the reader's, off by default) every end somebody may go in by
// has its arrow; the end tapped, a portal picked to move, and the end under the pointer
// show their own way out and every way in. Laid over the drawing rather than in it, so the
// pointer moving draws no board.
function paintLinks() {
  const svg = document.getElementById('board');
  if (!svg || !SESSION || !FRAME || typeof svg.querySelector !== 'function') return;
  const old = svg.querySelector('[data-link]');
  if (old) old.remove();
  const portals = SESSION.ruleset.portals;
  if (!portals) return;
  const ways = new Map();
  const add = (from, to) => {
    if (from && to && (from.open || []).some(Boolean)) ways.set(`${from.type}>${to.type}`, [from, to]);
  };
  if (LINKS) for (const end of portals.ends) add(end, portalPartner(portals, end));
  const own = new Set();
  if (LINKS && HOVER_END !== null) own.add(HOVER_END);
  if (INSPECT !== null) own.add(INSPECT);
  if (SELECTED !== null && (portalOn(SELECTED) || foldOn(SELECTED))) own.add(SELECTED);
  for (const square of own) {
    const here = portalOn(square) || foldOn(square);
    if (!here) continue;
    add(here.end, here.partner);
    for (const source of portalSources(portals, here.end)) add(source, here.end);
    // A payload from before schema 21 names no sources: a pair's partner leads back.
    if (here.partner && here.end.dest === undefined) add(here.partner, here.end);
  }
  const cell = 100 / Math.max(SESSION.ruleset.width, SESSION.ruleset.height);
  let out = '';
  for (const [from, to] of ways.values()) {
    const a = portalSquare(from.type), b = portalSquare(to.type);
    if (a < 0 || b < 0) continue;
    const [ax, ay] = boardCentre(a, cell);
    const [bx, by] = portalNearestCentre(b, ax, ay, cell, FRAME);
    out += portalArrowHtml(ax, ay, bx, by, cell, from.pair, `${from.type}>${to.type}`);
  }
  if (!out) return;
  const group = document.createElementNS(SVGNS, 'g');
  group.setAttribute('data-link', '');
  group.setAttribute('pointer-events', 'none');
  group.innerHTML = out;
  svg.appendChild(group);
}

// The keyboard on a board played inside the designer (*play it here*, P3's D9): the board
// takes the focus, arrow keys move a ring from square to square as the board is drawn,
// Enter or space is a click on that square -- picking a piece, playing a move, or opening
// the choice of where a piece comes out of a portal -- and Esc puts a picked piece down.
// Only there: on the board's own page the bare arrows step through the game, and that is
// the more common reading.
let BOARD_FOCUS = null;

function boardKeys(event) {
  if (!SESSION || PROMO || !BOARD_INLINE) return;
  const rules = SESSION.ruleset;
  const W = rules.width, H = rules.height;
  const key = event.key;
  const handled = () => { event.preventDefault(); event.stopPropagation(); };
  if (key === 'Enter' || key === ' ') {
    if (BOARD_FOCUS === null) return;
    handled();
    clickSquare(BOARD_FOCUS);
    paintBoardFocus();
    return;
  }
  if (key === 'Escape') {
    if (SELECTED === null) return;
    handled();
    SELECTED = null;
    drawBoard();
    return;
  }
  const by = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[key];
  if (!by) return;
  handled();
  // Up on the screen is up the board, or down it on a board turned round.
  const turn = FLIPPED ? -1 : 1;
  const absent = new Set(SESSION.absent || []);
  let at = BOARD_FOCUS;
  if (at === null) {
    at = SELECTED !== null ? SELECTED : Math.floor(H / 2) * W + Math.floor(W / 2);
  } else {
    let file = at % W, rank = Math.floor(at / W);
    for (let tries = 0; tries < Math.max(W, H); tries++) {
      file += by[0] * turn;
      rank += by[1] * turn;
      if (file < 0 || file >= W || rank < 0 || rank >= H) { file -= by[0] * turn; rank -= by[1] * turn; break; }
      if (!absent.has(rank * W + file)) break;
    }
    at = rank * W + file;
  }
  BOARD_FOCUS = at;
  paintBoardFocus();
}

// The keyboard's ring, laid over the drawing as the pair link is.
function paintBoardFocus() {
  const svg = document.getElementById('board');
  if (!svg || !SESSION || !FRAME || typeof svg.querySelector !== 'function') return;
  const old = svg.querySelector('[data-kbfocus]');
  if (old) old.remove();
  if (BOARD_FOCUS === null || !BOARD_INLINE) return;
  const cell = 100 / Math.max(SESSION.ruleset.width, SESSION.ruleset.height);
  const [x, y] = squareXY(BOARD_FOCUS, cell, FRAME);
  const group = document.createElementNS(SVGNS, 'g');
  group.setAttribute('data-kbfocus', String(BOARD_FOCUS));
  group.setAttribute('pointer-events', 'none');
  group.innerHTML = cellShape(x, y, cell, FRAME, `fill="none" stroke="var(--focus)" stroke-width="${
    cell * 0.06}"`, 0.04);
  svg.appendChild(group);
  svg.setAttribute('aria-label', `the board; the keyboard is on ${squareName(BOARD_FOCUS % SESSION.ruleset.width,
    Math.floor(BOARD_FOCUS / SESSION.ruleset.width))}: arrow keys move, Enter picks or plays, Esc puts a piece down`);
}

// The exit the chooser's focused or hovered button names, lit on the board (D6). On a
// wrapping board the lit copy is the one drawn nearest the end it comes out beside (N6):
// on a view slid across the seam that may be a ghost, and the square inside the outline
// is then the far side of the board from where the piece will be seen to come out.
function lightExit(move) {
  LIT = move || null;
  const svg = document.getElementById('board');
  if (!svg || typeof svg.querySelector !== 'function') return;
  const old = svg.querySelector('[data-lit]');
  if (old) old.remove();
  if (!LIT || !SESSION || !FRAME || LIT.portalOut === undefined) return;
  const cell = 100 / Math.max(SESSION.ruleset.width, SESSION.ruleset.height);
  const copy = nearestCopy(LIT.to, squareCell(LIT.portalOut, FRAME), FRAME);
  const [x, y] = cellXY(copy.u, copy.v, cell, FRAME);
  const group = document.createElementNS(SVGNS, 'g');
  group.setAttribute('data-lit', String(LIT.to));
  group.setAttribute('data-copy', `${copy.u},${copy.v}`);
  group.setAttribute('pointer-events', 'none');
  group.innerHTML = cellShape(x, y, cell, FRAME, 'fill="var(--hl-sel)" opacity="0.5"')
    + cellShape(x, y, cell, FRAME, `fill="none" stroke="var(--hl-sel)" stroke-width="${
      cell * 0.08}"`, 0.04);
  svg.appendChild(group);
}

// How big a cell is drawn, which is how much of a portal's glyph it keeps (D9): the
// board's `data-portals` tier, read by the stylesheet, and the pair's number held at
// `PORTAL_NUMBER_LEAST` px however small the cell, its dark eye grown to hold it.
function fitPortals(svg) {
  if (!svg || !SESSION || !SESSION.ruleset.portals) return;
  if (typeof svg.getBoundingClientRect !== 'function') return;
  const box = String(svg.getAttribute('viewBox') || '').split(/\s+/).map(Number);
  const width = svg.getBoundingClientRect().width;
  if (box.length !== 4 || !(box[2] > 0) || !(width > 0)) return;
  const cell = 100 / Math.max(SESSION.ruleset.width, SESSION.ruleset.height);
  const perUnit = width / box[2];
  svg.setAttribute('data-portals', portalTier(cell * perUnit));
  // Q838h: on a phone the arms are drawn by mass, as at the mid size (`pg-mass`).
  if (portalPhone()) svg.setAttribute('data-phone', '');
  else if (typeof svg.removeAttribute === 'function') svg.removeAttribute('data-phone');
  const least = PORTAL_NUMBER_LEAST / perUnit;
  for (const text of svg.querySelectorAll('.pg-num')) {
    const em = Number(text.dataset.em || PORTAL_NUMBER) * cell;
    text.setAttribute('font-size', Math.max(em, least).toFixed(3));
  }
  for (const eye of svg.querySelectorAll('.pg-eye')) {
    const r = Number(eye.dataset.em || PORTAL_EYE) * cell;
    eye.setAttribute('r', Math.max(r, least * PORTAL_EYE_LEAST).toFixed(3));
  }
}

// Always a <g>, even for the letter fallback, so the animation has one element per piece
// to move and the two-node fallback does not have to be special-cased.
function drawPiece(piece, x, y, cell, square) {
  const cx = x + cell / 2, cy = y + cell / 2;
  // An icon pair is one drawing per player and has no third entry, so a wild piece
  // (owner 2, schema 9) is always the lettered disc, in the third tint.
  // Generation 42 (N4c): the duck is nobody's too, but it is drawn as its type's own drawing
  // (the first seat's), since there is one duck and its picture is the rule's.
  const wild = piece.player > 1;
  const duck = wild && piece.type === SESSION.ruleset.duck;
  const pair = wild && !duck ? null : (SESSION.ruleset.icons || [])[piece.type];
  const icon = pair ? pair[duck ? 0 : piece.player] : null;
  const tag = square === undefined ? '' : ` data-piece="${square}"`;
  // Schema 19: a portal end is drawn by the board (`portalGlyph`, the vortex of Q838b): its
  // pair's colour and, where there are several, its number; whose pieces may go in, by its
  // arms; who may move it, by what it stands on. Its words go on the glyph for a screen
  // reader. The end on its own square turns, idly; a ghost copy of it, and the copy a
  // move's animation carries, stand still, so a board turns one swirl per end.
  // Schema 21 (Q843b): its beak points at the end it leads to as drawn, the nearest copy on
  // a wrapping board, from wherever this copy is drawn; and it is dimmed where the piece
  // picked may not go in by it.
  const portal = portalEnd(piece.type);
  if (portal) {
    const portals = SESSION.ruleset.portals;
    const said = square === undefined ? null : portalOn(square);
    const to = portalPartner(portals, portal);
    const there = to && FRAME ? portalSquare(to.type) : -1;
    return portalGlyph(portal, x, y, cell, {
      numbered: portalPairs(portals) > 1, label: said ? said.words.label : '',
      idle: square !== undefined, flipped: FLIPPED,
      aim: FRAME ? portalAimFor(portal, there, x, y, cell, FRAME) : null,
      dim: portalRefuses(portal),
      attrs: `${tag.trim()} pointer-events="none"` });
  }
  let inner;
  if (icon) {
    const size = cell * 0.88, corner = (cell - size) / 2;
    // A piece with a facing is drawn pointing where it faces, and nothing else turns.
    // `directional` names the types that have one; `facingAngles` gives the heading,
    // that table rather than `facing * 90`, because a mirrored army turns the other
    // way round on screen. The one other rotation is ownership: a single-colour set
    // (shogi) has both armies in one colour and marks the second by turning the tile.
    const heading = (SESSION.ruleset.directional || [])[piece.type]
      ? (SESSION.ruleset.facingAngles || [])[piece.player] : null;
    const angle = heading ? heading[piece.facing || 0] || 0 : (icon.rotate ? 180 : 0);
    const spin = angle ? ` transform="rotate(${angle} ${cx} ${cy})"` : '';
    inner = `<image href="${esc(icon.href)}" x="${x + corner}" y="${y + corner}" width="${size}"
             height="${size}" preserveAspectRatio="xMidYMid meet"${spin}/>`;
  } else {
    const fill = piece.player === 0 ? 'var(--p0)'
               : piece.player === 1 ? 'var(--p1)' : 'var(--p2)';
    const ink = piece.player === 0 ? 'var(--p0-ink)'
              : piece.player === 1 ? 'var(--p1-ink)' : 'var(--p2-ink)';
    inner = `<circle cx="${cx}" cy="${cy}" r="${cell * 0.36}" fill="${fill}" stroke="var(--ink)"
             stroke-width="${cell * 0.025}" opacity="0.95"/>
            <text x="${cx}" y="${cy}" fill="${ink}" font-size="${cell * 0.42}" font-weight="600"
             text-anchor="middle" dominant-baseline="central"
             >${esc(SESSION.ruleset.pieceIds[piece.type])}</text>`;
  }
  return `<g${tag} pointer-events="none">${inner}</g>`;
}

// How far the view has been shifted across the seam, in whole squares. A wrapping board
// has no boundary of its own, so this is the reader's chosen frame of reference and the
// outline follows it. Zero is the frame the squares are named in and nothing else.
let SHIFT = { files: 0, ranks: 0 };
// The frame the current drawing was made in. The animation has to place a square the
// same way the drawing did, so both read this rather than working it out twice.
let FRAME = null;

function shiftView(files, ranks) {
  SHIFT = { files: SHIFT.files + files, ranks: SHIFT.ranks + ranks };
  drawBoard();
}

function resetView() {
  SHIFT = { files: 0, ranks: 0 };
  drawBoard();
}

// Whether the promotion zone is drawn. The reader's, not the game's, so it is read out
// of the browser once and survives opening another board.
let ZONES = zonesShown();

// The toggle, at the end of the legend line under the board's controls, beside the words for the
// zone it puts away. A game with no promotion gets no button at all, which is how the
// control says what it is about.
function drawMarks() {
  const row = document.getElementById('marks');
  if (!row || !SESSION) return;
  // Schema 19: a game with portals also offers the pair link (§4), off until asked for.
  const zones = !!SESSION.ruleset.zones, links = !!SESSION.ruleset.portals;
  if (!zones && !links) { row.innerHTML = ''; row.dataset.built = ''; showLegendRow(); return; }
  const state = (zones ? (ZONES ? 'on' : 'off') : '') + (links ? (LINKS ? '/links' : '/') : '');
  if (row.dataset.built !== state) {
    row.innerHTML = (zones ? zoneToggleHtml(ZONES) : '') + (links ? linkToggleHtml(LINKS) : '');
    row.dataset.built = state;
    if (zones) document.getElementById('zonetoggle').onclick = toggleZones;
    if (links) document.getElementById('linktoggle').onclick = toggleLinks;
  }
  showLegendRow();
}

// The legend line takes no room while it has nothing to say: no marks and no toggle.
function showLegendRow() {
  const row = document.getElementById('legendRow');
  const legend = document.getElementById('legend');
  const marks = document.getElementById('marks');
  if (!row || !legend || !marks) return;
  row.hidden = !legend.innerHTML && !marks.innerHTML;
}

// What the marks on this board mean, in one line, and only the kinds the board is
// actually carrying. A ring is not self-explanatory and there is nowhere on a board to
// ask what one is for: a player who does not know that the teal ring is the square that
// wins the game is playing a different game from the one the rules describe. It is the
// frame's fourth slot, under the control bar (plan-visual.md §3.3), and `fitBoard` sizes
// the board so it is in the first screen: below the screen, where it was, every
// first-time reader misread the zone (study F1).
//
// The kinds come from the marks the drawing was given, so a kind that stops being drawn
// -- the promotion zone, which the reader can put away -- leaves the line with it, and a
// board with no marks at all gets no line. Each entry is its `MARK_KINDS` row: the key
// drawn by the same code that draws the mark, then its name and what it means.
//
// `drawn` names the kinds drawn between squares rather than on one -- a wild piece's coming
// step -- which the marks map cannot carry.
function drawLegend(marks, drawn) {
  const row = document.getElementById('legend');
  if (!row || !SESSION) return;
  const kinds = new Set(drawn || []);
  for (const square of Object.keys(marks || {})) {
    for (const mark of marks[square]) kinds.add(mark.kind);
  }
  // A seam is drawn on the board's edges rather than on a square, so it is read off the
  // rules the board was drawn from, and then named from its row like every other kind.
  const said = legendEntries(kinds, SESSION.ruleset.wrap, zoneWords(SESSION.ruleset.zones),
                             SESSION.ruleset.portals);
  // Schema 28 (Q943b, `docs/archive/plan-speed.md` §4.5): a game with speeds lists its ranks, which
  // no square shows.
  const speeds = typeof speedLegendHtml === 'function'
    ? speedLegendHtml(SESSION.ruleset.speed, SESSION.ruleset.pieceNames) : '';
  if (speeds) said.push(speeds);
  row.innerHTML = said.length ? 'on the board: ' + said.join(' ') : '';
  showLegendRow();
}

// The legend's entries for a set of mark kinds, one per `MARK_KINDS` row drawn, in the
// row's own words: the play board's line above and the game page's start position (K1)
// both say them from here, so a mark has one name wherever it is drawn. A seam is read
// off the rules' `wrap`, since it lies on the board's edges rather than on a square;
// `zoneMore` is the zone's clause where the caller has one; `portals` the game's
// `portal_payload`, whose first end is the portal's key. A row with a `gloss` carries its
// (i) after its words: the portal's draws four ends with their sentences (Q838c).
function legendEntries(kinds, wrap, zoneMore, portals) {
  const seams = seamAxes(wrap);
  if (seams.files || seams.ranks || cutSeams(wrap).barrier.length) kinds.add('seam');
  const said = [];
  for (const kind of Object.keys(MARK_KINDS)) {
    if (!kinds.has(kind)) continue;
    const look = MARK_KINDS[kind];
    const more = kind === 'zone' ? zoneMore || '' : '';
    // A portal's key is the game's first momentum end, a fold's its first fold (schema 22).
    const sample = (kind === 'portal' || kind === 'fold') && portals && portals.ends
      ? portals.ends.find(end => isFold(end) === (kind === 'fold')) : undefined;
    const gloss = look.gloss && typeof info === 'function' ? ' ' + info(look.gloss) : '';
    said.push(`<span class="entry" data-legend="${kind}">${markKey(kind, sample)}<b
      class="${styleClass(`color:${look.ink || look.colour}`)}">${esc(look.name)}</b> ${esc(look.label)}${
      esc(more)}${gloss}</span>`);
  }
  return said;
}

// Whose zone is whose, now the zone has no beads: each side's band named by its ranks,
// read off the compiled zones (`zone_payload`) and named as the board labels them. Said
// only where a side's zone is whole ranks, which is the only shape "rank 8" describes;
// a zone both sides share, or one cut some other way, gets no clause at all rather than
// a wrong one.
function zoneWords(zones) {
  if (!zones || !SESSION) return '';
  const W = SESSION.ruleset.width;
  const first = zones[0] || [], second = zones[1] || [];
  if (first.length === second.length && first.every((square, i) => square === second[i])) {
    return '';
  }
  const absent = new Set(SESSION.absent || []);
  const ranksOf = squares => {
    const ranks = [...new Set(squares.map(square => Math.floor(square / W)))].sort((a, b) => a - b);
    // Whole ranks only: every square of every rank named is in the zone.
    for (const rank of ranks) {
      for (let file = 0; file < W; file++) {
        const square = rank * W + file;
        if (!absent.has(square) && !squares.includes(square)) return null;
      }
    }
    return ranks.length ? ranks : null;
  };
  const words = (squares, player) => {
    const ranks = ranksOf(squares);
    if (!ranks) return null;
    const names = ranks.map(rank => squareName(0, rank).slice(1));
    const contiguous = ranks.every((rank, i) => i === 0 || rank === ranks[i - 1] + 1);
    const span = names.length === 1 ? `rank ${names[0]}`
      : contiguous ? `ranks ${names[0]}-${names[names.length - 1]}`
        : `ranks ${names.join(', ')}`;
    return `${sideName(player)}'s on ${span}`;
  };
  const both = [words(first, 0), words(second, 1)];
  if (both.some(said => !said)) return '';
  return ` (${both.join(', ')})`;
}

function toggleZones() {
  ZONES = !ZONES;
  rememberZones(ZONES);
  drawBoard();
}

// Moving the view across the seam, one square at a time. There is no boundary to be at
// the middle of, so these say which repeat is on screen rather than where the board is;
// `reset` goes back to the frame the squares take their names from.
//
// Only the axes that wrap get buttons, and a board with edges gets no row at all. The
// keys are shift and an arrow, because the bare arrows step through the game already.
function drawShift() {
  const row = document.getElementById('shift');
  if (!row || !SESSION) return;
  const wrap = wrapAxes(SESSION.ruleset.wrap);
  if (!wrap.files && !wrap.ranks) { row.innerHTML = ''; row.dataset.built = ''; return; }
  if (row.dataset.built !== 'yes') {
    const across = wrap.files ? `
      <button data-shift="-1,0" title="one file back (shift + left arrow)">&larr;</button>
      <button data-shift="1,0" title="one file on (shift + right arrow)">&rarr;</button>` : '';
    const up = wrap.ranks ? `
      <button data-shift="0,1" title="one rank up (shift + up arrow)">&uarr;</button>
      <button data-shift="0,-1" title="one rank down (shift + down arrow)">&darr;</button>` : '';
    row.innerHTML = `<span class="muted small">slide the board</span>${across}${up}
      <button id="reframe" title="back to the normal view">reset</button>
      <span class="muted small" id="frame"></span>`;
    row.dataset.built = 'yes';
    row.querySelectorAll('[data-shift]').forEach(node => {
      const by = node.dataset.shift.split(',');
      node.onclick = () => shiftView(Number(by[0]), Number(by[1]));
    });
    document.getElementById('reframe').onclick = resetView;
  }
  // Which square the repeat on screen runs from, read off the frame rather than off
  // `SHIFT` -- eight clicks along an eight-file board is back where it started.
  const corner = FRAME ? cellSquare(0, 0, FRAME) : 0;
  const W = SESSION.ruleset.width;
  const repeat = 'showing from ' + squareName(corner % W, Math.floor(corner / W));
  const reframe = document.getElementById('reframe');
  reframe.disabled = corner === 0;
  // Also on the button, because a phone's control row leaves the sentence out.
  reframe.title = 'back to the normal view (now ' + repeat + ')';
  document.getElementById('frame').textContent = repeat;
}

function drawBoard(previous) {
  const rules = SESSION.ruleset;
  const W = rules.width, H = rules.height;
  const cell = 100 / Math.max(W, H);
  const svg = document.getElementById('board');
  if (!svg) return;

  // What the ply currently arriving is made of. Worked out here rather than inside the
  // animation because the *drawing* needs it too: the squares a wild piece walked
  // between are marked, and those marks have to be there for a reader who has turned
  // motion off and have to survive a redraw for a flip or a piece highlight.
  // Never while reviewing: a step *is* a position, and the review draw hands it over
  // whole rather than as a transition to be played.
  //
  // The one exception is `previous.review`: a step forward of one beat over a move through
  // a portal (Q836, `showBeat`), which plays that move's journey from the beat stepped
  // from, because a piece gone from one side of the board and standing on the other is
  // the one move a reader cannot follow from the two positions alone.
  const review = !!(previous && previous.review && AT);
  const beats = review ? [{ kind: 'move', board: AT.board, steps: null }]
    : (previous && !AT) ? plyBeats(SESSION.board, SESSION.lastMove) : null;
  if (beats && !review) PHASE = beats.length > 1 ? beats[1].steps : null;
  const walked = new Map();
  for (const step of (PHASE || [])) { walked.set(step.from, 'from'); walked.set(step.to, 'to'); }

  // The frame this drawing is made in. Everything below places a square through it, and
  // so does the animation, because a board and a piece flying across it that disagreed
  // about where a square is would send the piece to the square it used to be on.
  FRAME = boardView(W, H, wrapAxes(rules.wrap), SHIFT, FLIPPED, rules.lattice,
                    boardCut(SESSION.absent, rules.wrap));
  svg.setAttribute('viewBox', viewBox(FRAME, cell));

  // A move picked between two people and not sent yet (Q733), drawn on the game as it
  // stands and never on a position the reader has stepped back to.
  const held = AT ? null : corrHeldDrawn();
  // With a move held and nothing else picked, the held piece's other squares keep their
  // dots: a tap on one moves it there instead (Q740, `corrRepick`).
  const picked = SELECTED !== null ? SELECTED
    : held && DROP === null && held.from !== null ? held.from : null;
  const targets = new Set(movesFrom(picked).map(m => m.to));
  if (held && SELECTED === null) { targets.delete(held.to); targets.delete(held.dest); }
  // Schema 19: the picked piece's ways through a portal (`docs/archive/plan-portals.md` §4). Each
  // square it comes out on takes a dot in the colour of the pair it goes in by (`through`),
  // beside the plain dot where it can also get there directly (`direct`); each end it can
  // go in by is ringed, since a tap there is a way in (`entries`, D6).
  const through = new Map(), direct = new Set(), entries = new Set();
  for (const m of movesFrom(picked)) {
    // Schema 24 (`docs/archive/plan-portals-2.md` §5.3R, the user: *"as simple as selecting which
    // square to land on, just as any other landing square"*): a move through a fold ends on
    // the far end, so that square takes the ordinary dot, or ring, like any landing; staying
    // is the ordinary dot on the fold. No hue, no chooser: the two targets differ.
    if (m.portal === undefined) { direct.add(m.to); continue; }
    const way = portalOn(m.portal);
    through.set(m.to, way ? `var(--pair-${way.end.pair})` : 'var(--dot)');
    entries.add(m.portal);
  }
  const dropTargets = DROP === null ? new Set()
    : new Set(SESSION.moves.filter(m => m.drop === DROP).map(m => m.to));
  // Generation 42 (N4c, `docs/plan-npc.md` §4.6): with the duck owed, every square it may go
  // to is lit at once, with no first click on the duck, and its own square is ringed.
  const duckLit = duckTargets();
  const duckRing = duckMode() ? SESSION.duck : null;
  const last = AT ? AT.lastMove : SESSION.lastMove;
  const colour = tint();
  const walls = new Set(SESSION.obstacles || []);
  const folds = new Map(foldsShown().map(fold => [fold.square, fold]));
  const groups = rules.portals ? portalPairs(rules.portals) : 0;
  const lit = HIGHLIGHT === null ? null : HIGHLIGHT;

  // Squares the rules single out, a list per square: the ones a game ends on when a
  // piece reaches one, and the ones a piece promotes on. Built from the compiled rules
  // the session carries, so it belongs to the game on screen and cannot outlive it.
  // `ZONES` is the reader's own toggle, which is why it is read here on every draw.
  // The check is the position's, not the rules': the square of the royal piece the engine
  // said is attacked (`checkSquares`).
  const SQUARE_MARKS = squareMarks(Object.assign({}, rules, { checked: checkSquares() }),
                                   { zone: !ZONES });
  // Each square its own piece of wood, cut at an offset hashed from the game and the
  // square (`wood.js`), so the same game is grained the same on every draw.
  const identity = rules.hash || rules.name || `${W}x${H}`;
  const wood = woodGrain('board', identity, cell);
  const framed = svg.closest ? svg.closest('.board-frame') : svg.parentNode;
  if (framed && framed.style && typeof framed.style.setProperty === 'function') {
    const at = woodFrameAt(identity);
    framed.style.setProperty('--wood-x', at[0] + 'px');
    framed.style.setProperty('--wood-y', at[1] + 'px');
  }

  // Where the squares' names go: outside the board where its drawing ends at its edge
  // (`drawCoords`), on the naming frame's own squares where ghosts or a seam's arrows
  // already stand round it.
  const outside = labelsOutside(FRAME);

  // One square, wherever in the grid it is being drawn. `depth` is zero inside the
  // outline and says how far out a ghost copy lies otherwise.
  //
  // A ghost is the same square again, so it draws the same things -- the last move's
  // mark, a selection, the dots saying where a selected piece may go -- and a click on
  // one is a click on that square. Two things it does not take: the `data-piece` tag,
  // which the animation looks a piece up by and which has to name one node, and the
  // coordinate label, which belongs to the frame of reference the outline marks.
  function cellHtml(square, u, v, depth) {
    const corner = cellXY(u, v, cell, FRAME);
    const x = corner[0], y = corner[1];
    const file = square % W, rank = Math.floor(square / W);
    const real = depth === 0;
    const shade = cellShade(file, rank, FRAME);
    // A portal end's words are its square's tooltip (D9), on the square itself because the
    // glyph, like every piece, takes no pointer.
    const portal = portalOn(square) || foldOn(square);
    const base = cellShape(x, y, cell, FRAME, `fill="${shade}" data-square="${square}"`);
    let out = (portal ? base.replace(/\/>$/, `><title>${esc(portal.words.label)}</title></${
      FRAME.hex ? 'path' : 'rect'}>`) : base) + wood.cell(x, y, FRAME, square, shade);
    if (walls.has(square)) {
      out += cellShape(x, y, cell, FRAME, 'pointer-events="none" fill="var(--wall)"');
      out += `<path d="${wallCross(x, y, cell, FRAME)}"
               stroke="var(--ink)" stroke-width="${cell * 0.06}" opacity="0.5"
               pointer-events="none"/>`;
    }
    if (colour[square]) {
      out += cellShape(x, y, cell, FRAME, `pointer-events="none"
               fill="${colour[square] === 1 ? 'var(--claim0)' : 'var(--claim1)'}" opacity="0.3"`);
    }
    // Schema 22: a fold lies in the board, under any piece standing in it: the pool
    // (`foldGlyph`, Q843d, `docs/archive/plan-portals-2.md` §5.3), its beak aimed at the end it leads
    // to as drawn, its number a badge over the piece when one stands in it (below). The one
    // on its own square turns, idly; a ghost copy stands still and plain.
    const fold = folds.get(square);
    const foldEnd = fold ? portalEnd(fold.type) : null;
    if (foldEnd) {
      const there = portal && portal.there !== undefined ? portal.there : -1;
      out += foldGlyph(foldEnd, x, y, cell, {
        numbered: groups > 1, occupied: !!pieceAt(square), idle: real, plain: !real,
        flipped: FLIPPED, aim: portalAimFor(foldEnd, there, x, y, cell, FRAME),
        dim: portalRefuses(foldEnd), label: real && portal ? portal.words.label : '',
        attrs: `${real ? `data-fold-at="${square}" ` : ''}pointer-events="none"` });
    }
    // Under the piece, because a square that ends the game is a square pieces arrive
    // on, and the piece standing there has to stay the thing you see first.
    if (SQUARE_MARKS[square]) out += squareBadge(x, y, cell, SQUARE_MARKS[square], FRAME, 'board');
    // A move through a portal tints the two ends it passed as well (§4: the trail in two
    // legs, origin to entry and exit to landing, its pair's two cells marked); the legs
    // themselves are drawn over the board (`portalLegsHtml`).
    // A round (schema 26) tints both players' moves: `lastMove` is its second entry, and
    // its `reveal` holds both.
    if (last && (square === last.from || square === last.to
        || (last.portal !== undefined && (square === last.portal || square === last.portalOut))
        || (last.foldIn !== undefined && (square === last.foldIn || square === last.foldOut))
        || (last.reveal && last.reveal.moves.some(m => m.from === square || m.to === square)))) {
      out += mark(x, y, cell, 'var(--hl-last)', 0.34);
    }
    // Where the wild piece walked, in its own hand and not the mover's: a dashed
    // outline rather than a tint, because the palette's third owner is an amber a
    // shade off the last-move mark and "the boar did this" has to be legible as a
    // *different* kind of event, not a slightly different colour. It stays on the
    // board after the motion has finished, which is what a reader who looked away
    // for a second needs in order to see what took the piece that is now missing.
    if (walked.has(square)) out += trail(x, y, cell, walked.get(square) === 'to');
    if (square === SELECTED) out += mark(x, y, cell, 'var(--hl-sel)', 0.55);
    const piece = pieceAt(square);
    if (piece) {
      if (lit !== null && piece.type === lit) out += mark(x, y, cell, 'var(--hl-sel)', 0.45);
      // The piece a held move lifts, and the one it would take, stay where they are,
      // faded: the move is not made until it is sent.
      const lifted = held && (square === held.from || square === held.to || square === held.dest);
      const drawn = drawPiece(piece, x, y, cell, real ? square : undefined);
      out += lifted ? `<g opacity="${HELD_FADE}">${drawn}</g>` : drawn;
      // The fold's number, moved off the piece standing in it to the cell's corner.
      if (foldEnd && groups > 1) out += foldBadgeHtml(foldEnd, x, y, cell);
    }
    if (held && (square === held.from || square === held.dest)) {
      out += heldOutline(x, y, cell, square === held.dest);
    }
    const way = through.get(square);
    if (way && !direct.has(square) && !piece) {
      // Reached only through a portal: the dot in the pair's colour, edged so it holds on
      // both woods.
      out += `<circle cx="${x + cell / 2}" cy="${y + cell / 2}" r="${cell * 0.13}"
               fill="${way}" stroke="var(--portal-edge)" stroke-width="${cell * 0.03}"
               data-square="${square}" data-through="1"/>`;
    } else if (targets.has(square) || dropTargets.has(square) || duckLit.has(square)) {
      out += `<circle cx="${x + cell / 2}" cy="${y + cell / 2}"
               r="${piece ? cell * 0.44 : cell * 0.13}" fill="${piece ? 'none' : 'var(--dot)'}"
               stroke="var(--dot)" stroke-width="${cell * 0.05}" opacity="0.8"
               data-square="${square}"/>`;
      // And through a portal too, or a capture there: a small dot of the pair's colour on
      // the square's corner, clear of the piece.
      if (way) {
        out += `<circle cx="${x + cell * 0.8}" cy="${y + cell * 0.2}" r="${cell * 0.09}"
                 fill="${way}" stroke="var(--portal-edge)" stroke-width="${cell * 0.025}"
                 data-square="${square}" data-through="1"/>`;
      }
    }
    if (real && square === duckRing) {
      out += cellShape(x, y, cell, FRAME, `fill="none" stroke="var(--held-halo)"
               stroke-width="${cell * 0.12}" opacity="0.9" pointer-events="none"`, 0.06)
        + cellShape(x, y, cell, FRAME, `fill="none" stroke="var(--duck)"
               stroke-width="${cell * 0.065}" pointer-events="none" data-duck-ring="1"`, 0.06);
    }
    if (entries.has(square)) {
      out += `<circle cx="${x + cell / 2}" cy="${y + cell / 2}" r="${cell * 0.47}" fill="none"
               stroke="var(--dot)" stroke-width="${cell * 0.035}" stroke-dasharray="${
               cell * 0.09} ${cell * 0.06}" opacity="0.8" data-square="${square}"/>`;
    }
    if (real) return outside ? out : out + coordsHtml(square, x, y, cell, FRAME);
    return `<g opacity="${ghostFade(depth).toFixed(3)}">${out}</g>`;
  }

  // The board continuing past the outline first, then the repeat the reader is in, then
  // the outline over both. A board with edges has no ghosts and no outline, so it comes
  // out of this exactly as it always did.
  let out = '';
  for (const ghost of ghostCells(FRAME)) {
    out += cellHtml(ghost.square, ghost.u, ghost.v, ghost.depth);
  }
  for (let rank = H - 1; rank >= 0; rank--) {
    for (let file = 0; file < W; file++) {
      const square = rank * W + file;
      // A hexagon's corners are not cells: no path, no hatch, nothing to click.
      if (cellAbsent(square, FRAME)) continue;
      const place = squareCell(square, FRAME);
      out += cellHtml(square, place[0], place[1], 0);
    }
  }
  // The outline, then the seam over it. The outline stands down on an axis whose seam
  // is a fact of the game, so a reader is never offered two boundaries on one axis with
  // only one of them meaning anything.
  // The glow each pool spills onto the squares round it, over them (a later square would
  // cover it), clear of the piece standing in the pool.
  for (const fold of foldsShown()) {
    if (!portalEnd(fold.type) || cellAbsent(fold.square, FRAME)) continue;
    const place = squareCell(fold.square, FRAME);
    const corner = cellXY(place[0], place[1], cell, FRAME);
    out += foldGlowHtml(corner[0], corner[1], cell, fold.pair);
  }
  out += outlineHtml(cell, FRAME, rules.wrap) + barrierHtml(cell, FRAME, rules.wrap);
  // Where each wild piece steps next, over everything: a dotted arrow from the edge of
  // the piece to short of the square's middle, so it never covers the piece or a
  // legal-move dot; and, fainter, where it steps after the move after that (Q764). The
  // core's answer for the position on screen (`coming`, `comingThen`).
  const coming = AT ? AT.coming : SESSION.coming;
  const comingThen = AT ? AT.comingThen : SESSION.comingThen;
  out += comingHtml(coming, cell, W, H, FRAME, rules.wrap, comingThen);
  // Schema 26: two pieces that can only take each other by a capture-only move, linked
  // after a void round or two (`standoffHtml`, the core's `voidRun` and `bars`).
  const standing = AT ? null : SESSION.round;
  out += standoffHtml(standing, cell);
  // Schema 27 (Q942b): a swap that met, between its two squares, while it is the last
  // round on screen; and (schema 29) each resting piece's badge, in any game, on the position
  // on screen (a step back's from its move, Q983b), numbered with its turns left where some
  // piece rests longer than one (`restNumbered`). Nothing rests once the game is over.
  const clashed = roundClash(last);
  out += clashHtml(clashed, cell);
  const resting = SESSION.outcome && !AT ? [] : positionRests(AT ? AT.resting : SESSION.resting);
  out += restsHtml(resting, cell, restNumbered(rules.rest));
  // Schema 30 (Q999c, `docs/archive/plan-passive.md` §4.6): each piece under a line of sight's
  // effect wears its badge, off the core's effect bits for the position on screen.
  const affected = effectsShown();
  out += effectsHtml(affected, cell);
  // Schema 28 (Q943b): which of the last round's two moves was made first, between speeds.
  const ordered = roundOrder(last);
  out += ordersHtml(ordered, cell);
  // Schema 27 (Q935c): the move announced in a spy game's announced round, shown to both
  // players, under the answer being held.
  const announced = SESSION.outcome ? null : announcedArrow(standing);
  out += announcedHtml(announced, cell, W, H, rules.wrap);
  // The held move over everything: its arrow, then its piece where it would land.
  if (held) out += heldHtml(held, cell, W, H, rules.wrap);
  // A move through a portal, in its two legs (§4), over the squares it tinted.
  out += portalLegsHtml(last, cell, W, H) + foldLegsHtml(last, cell, W, H);
  // Generation 42 (N4c): the duck's flight on the ply on screen, its own arrow.
  const flight = duckFlightHtml(last, cell);
  out += flight;

  svg.innerHTML = markDefs(SQUARE_MARKS, 'board') + wood.defs() + out;
  // A piece is drawn `pointer-events="none"`, the held move's faded one too, so a tap on
  // it is a tap on its square: on the held move's landing square, which takes it back.
  svg.onclick = event => {
    const node = event.target.closest('[data-square]');
    if (node) clickSquare(Number(node.dataset.square));
  };
  // The pair link under the pointer (§4), while the reader has links on.
  svg.onpointerover = event => {
    const node = event.target.closest ? event.target.closest('[data-square]') : null;
    const square = node ? Number(node.dataset.square) : null;
    const hover = portalOn(square) || foldOn(square) ? square : null;
    if (hover !== HOVER_END) { HOVER_END = hover; paintLinks(); }
    // Schema 30: a piece with a line of sight, under the pointer, shows what it sees.
    const looking = square !== null && sightOf(square) ? square : null;
    if (looking !== SIGHT_HOVER) { SIGHT_HOVER = looking; paintSight(); }
  };
  svg.onpointerleave = () => {
    if (HOVER_END !== null) { HOVER_END = null; paintLinks(); }
    if (SIGHT_HOVER !== null) { SIGHT_HOVER = null; paintSight(); }
  };
  // A board inside the designer takes the keyboard (D9); the board's own page does not.
  if (BOARD_INLINE && typeof svg.setAttribute === 'function') {
    svg.setAttribute('tabindex', '0');
    if (!svg.getAttribute('aria-label')) {
      svg.setAttribute('aria-label', 'the board: arrow keys move from square to square, Enter'
        + ' picks or plays, Esc puts a piece down');
    }
    svg.onkeydown = boardKeys;
  }

  // Redrawing replaces the board's contents outright, so whatever the last move was
  // still animating is destroyed by the next draw. `BUSY_UNTIL` is when it would have
  // finished, and `whenSettled` is what respects it.
  BUSY_UNTIL = beats
    ? performance.now() + animateChange(svg, previous, cell, W, H, FRAME, beats,
                                        review ? AT.lastMove : SESSION.lastMove) : 0;
  const busy = BUSY_UNTIL ? BUSY_UNTIL - performance.now() : 0;
  // What is coming is about the position the ply *leaves*, so it shows once the ply has
  // finished arriving: held back while the pieces move, then faded in.
  const arrows = svg.querySelector('[data-coming]');
  if (arrows && busy > 0 && typeof arrows.animate === 'function') {
    arrows.animate([{ opacity: 0 }, { opacity: 1 }],
                   { duration: PACE.beat, delay: busy, fill: 'backwards' });
  }
  // What the *game* is showing, which a review draw does not change: the next ply has
  // to animate from the position before it whether or not somebody was looking at move
  // six when it arrived.
  if (!AT) SHOWN = { board: SESSION.board.slice(), colour: (SESSION.colour || []).slice() };

  drawHand();
  drawStatus();
  drawPieceSays();
  drawReview();
  drawShift();
  drawMarks();
  drawLegend(SQUARE_MARKS, [
    ...(coming && coming.length ? ['coming'] : []),
    ...(comingThen && comingThen.length ? ['comingThen'] : []),
    ...boardKinds(SESSION.obstacles, SESSION.ruleset.portals),
    ...standoffKinds(standing),
    ...(clashed ? ['clash'] : []),
    ...(resting.length ? ['rested'] : []),
    ...effectLegendKinds(affected),
    ...(ordered.length ? ['order'] : []),
    ...(announced ? ['announced'] : []),
    ...(flight ? ['duckFlight'] : []),
  ]);
  drawCoords(cell, outside);
  fitBoard();
  fitPortals(svg);
  // The pair link and a lit exit live over the drawing, so they are laid again on it, and
  // so does what a piece sees (schema 30).
  paintLinks();
  paintSight();
  lightExit(LIT);
  paintBoardFocus();
  // The choice of what a piece becomes belongs to the position it was asked on: a step
  // back through the game, a take-back or an answer that moved the game puts it away.
  if (PROMO && !promotionStands()) closePromotion();
}

// The squares' names in a margin of their own beside the board (plan-visual.md §3.3), in
// the numbers' face, where no piece and no mark covers them. Each is placed where
// `edgeLabels` says its edge is, as a share of the drawing so it follows the board's size
// without a redraw, and set off outwards by the frame's wood and a little more.
function drawCoords(cell, outside) {
  const layer = document.getElementById('coords');
  if (!layer) return;
  const panel = layer.closest ? layer.closest('.board-panel') : null;
  if (panel && panel.classList) panel.classList.toggle('coords-out', !!outside);
  // A hex board's names fall in the corners its cells leave bare, on the frame's wood, so
  // each is set close to its own cell's edge, on a scrap of the page's paper.
  layer.classList.toggle('on-wood', !!(outside && FRAME.hex));
  if (!outside) { layer.innerHTML = ''; return; }
  layer.innerHTML = edgeNamesHtml(cell, FRAME, viewBox(FRAME, cell));
}

// --------------------------------------------------------------- the animation
//
// Driven by a diff of the two positions rather than by the move encoding, which is what
// makes it general: castling moves two pieces, en passant removes one that is not on the
// destination square, an atomic capture clears a neighbourhood, and a drop adds a piece
// that came from nowhere. None of the four is special-cased here, because a diff sees
// all of them as the same three facts -- something left a square, something arrived on
// one, something vanished.
//
// Every duration the board uses is stated here, in milliseconds, and nowhere else. It is
// deliberately unhurried -- a move is something a person has to *read*, and the pace this
// replaces was fast enough that a capture was finished before the capturing piece had
// arrived.
const PACE = {
  travelMin: 225,        // a move of no distance at all
  travelPerUnit: 21,     // ... and this much more per viewBox unit crossed
  travelMax: 630,        // ... but never longer than this, however wide the board
  impact: 0.8,           // the share of the travel after which the victim is struck
  strike: 390,           // a captured piece being knocked off the board
  burst: 330,            // the ring that marks where it was taken
  drop: 330,             // a piece arriving from a hand
  explosion: 480,        // the fiery burst that marks an explosion
  resist: 500,           // the shield shimmer on a piece that survived the blast
  rotate: 350,           // a piece turning to a new facing
  // A void move in a round (schema 26, Q918c): the piece steps toward the square it went
  // for and back, in this long, reaching this share of the way.
  voidStep: 460,
  voidReach: 0.42,
  // The gap between a ply's two beats -- the move somebody made, and the wild-piece
  // phase the rules run after it. Not a delay put in to hide a race: the phase happens
  // after the move, so it is *drawn* after the move, and this is only the pause
  // between them. Shorter than the shortest travel (225ms), so a machine game does not
  // crawl; long enough that the piece that moved has landed and come to rest before
  // anything else on the board starts moving, which is the whole of what was missing.
  beat: 200,
  // A move through a portal (schema 19, Q836, `docs/archive/plan-portals.md` *Travel through a
  // portal, animated*): one journey in three movements, laid out by `portalJourney`. The
  // slide to the end it goes in by, quicker than a travel because the portal pulls ...
  portalApproachMin: 170,
  portalApproachPerUnit: 7,
  portalApproachMax: 320,
  // ... the share of that slide after which the dive starts, so the piece is already
  // winding down into the rim as it arrives rather than stopping and then shrinking ...
  portalOverlap: 0.72,
  // ... the shrink and spin down into it ...
  portalDive: 250,
  // ... the instant it is nowhere, between the two ends ...
  portalVoid: 40,
  // ... and the grow and spin up out of the other end, the one step on, and the settle.
  portalEmerge: 400,
  // The share of the emergence after which a piece standing where it lands is struck:
  // the moment the piece, full size again, reaches that square.
  portalImpact: 0.6,
  // With reduced motion: no slide, no spin, no scale; the piece fades from where it was
  // and into where it lands, the two overlapping, in this long all told.
  portalStill: 300,
  // A move through a fold (schema 24, `docs/archive/plan-portals-2.md` §5.3R), laid out by
  // `foldJourney`: the slide onto the fold is the portal's approach; then the sink into the
  // pool (no spin: it goes through a surface, not down a drain) ...
  foldSink: 240,
  // ... the instant under the surface ...
  foldUnder: 60,
  // ... and the rise out of the far end, where the move ends.
  foldRise: 260,
  // A ring of ripple in the hue, where it sinks and where it rises.
  foldRipple: 520,
};
const SVGNS = 'http://www.w3.org/2000/svg';
let BUSY_UNTIL = 0;
// The wild-piece phase of the ply currently on screen -- `[{from, to, eaten}]`, or
// null. Kept beside SHOWN and for the same reason: the board is redrawn for a flip or a
// piece highlight, and the marks that say where the boar came from have to survive it.
let PHASE = null;

// A ply is beats: the mover's move, and then the wild-piece phase the rules run after it
// inside the same `make`. `plyBeats` splits one, in `viz/wrapped.py`, because the offline
// replay steps through the same beats this board animates.

// Returns how long the board will be busy, in milliseconds, so that the next draw can
// wait for it rather than wiping it.
//
// The two copies a move across the seam is drawn as. The piece arriving is the one the
// board already drew, flown in from outside the far edge; the piece leaving is a copy of
// what stood on the origin square, flown out through the near one. Both are the same
// translation, one repeat apart, and they hand over in the middle -- so what a reader
// follows is one piece going off the board and coming back on, which is what the rule
// says happens, rather than a piece sliding the wrong way across the middle.
//
// `flight` is how long the arriving piece takes, so the two are one motion however far
// the board says it went.
function animateCrossing(svg, node, was, fx, fy, dx, dy, cell, arc, flight, delay) {
  const start = delay || 0;
  node.animate(crossOpacity(false), { duration: flight, delay: start, fill: 'backwards' });
  if (!was) return;
  const away = ghostAt(svg, was, fx + dx, fy + dy, cell);
  travel(away, -dx, -dy, arc, start);
  const fade = away.animate(crossOpacity(true),
                            { duration: flight, delay: start, fill: 'both' });
  fade.onfinish = () => away.remove();
}

// One beat after the other, in the order the rules ran them. `last` is the move the
// beats are of: the game's last move, or the move a review step forward walks over.
function animateChange(svg, previous, cell, W, H, view, beats, last) {
  const before = previous.board;
  // The position the mover's own move leaves behind. For a ruleset with no wild pieces
  // that is the whole ply, and this is `SESSION.board`.
  const after = beats[0].board;
  if (!before || before.length !== after.length) return 0;
  // Reduced motion: nothing moves, as before, except that a trip through a portal gets
  // its calm version, because a piece that vanished on one side of the board and stands
  // on the other is the one move a snap leaves a reader unable to follow.
  if (STILL.matches) return portalStill(svg, before, after, last, cell, view);
  const phase = beats.length > 1 ? beats[1].steps : null;
  // Squares the phase moved onto: the piece the *page* draws there is the wild piece,
  // so anything the mover's beat puts there needs a ghost of its own.
  const swallowed = new Set(phase ? phase.map(step => step.to) : []);
  // Schema 26: a round is both players' moves at once, drawn from the core's account of it
  // (`animateRound`, `simultaneous.js`).
  const beat = last && last.reveal
    ? animateRound(svg, before, after, last.reveal, cell, W, H, view)
    : animateMove(svg, before, after, swallowed, cell, W, H, view, last);
  if (!phase) return beat.busy;
  return Math.max(beat.busy, animatePhase(svg, phase, after, beat, cell, W, H, view));
}

// The first beat: the move somebody submitted, diffed against the position that move
// alone leaves behind. Unchanged in every ruleset without a wild-piece phase, where
// that position is the whole ply.
//
// Reports what the phase needs from it: `landed`, when the last piece of this beat came
// to rest, and `doomed`, the ghosts it left standing on squares the phase is about to
// take them on.
function animateMove(svg, before, after, swallowed, cell, W, H, view, last) {
  // A piece the mover moved onto a square the phase then took it on. The board draws
  // the wild piece there, so this beat travels a ghost instead -- and the phase strikes
  // that same ghost, which is what makes "it moved there and was eaten" one story.
  const doomed = new Map();
  // When the last piece of *this* beat comes to rest, which is when the phase may start.
  let landed = 0, busy = 0;
  const nothing = () => ({ busy, landed, doomed });

  const gone = [], came = [];
  for (let sq = 0; sq < after.length; sq++) {
    if (before[sq] === after[sq]) continue;
    if (before[sq]) gone.push(sq);
    if (after[sq]) came.push(sq);
  }
  if (!gone.length && !came.length) return nothing();

  // Rotation: a piece that changed value on the same square without moving.
  // Detect it before the travel/capture logic, animate a spin, and remove the
  // square from gone/came so the rest of the pipeline ignores it.
  const rotated = [];
  for (let i = gone.length - 1; i >= 0; i--) {
    const sq = gone[i];
    const j = came.indexOf(sq);
    if (j === -1) continue;
    const oldP = pieceOf(before[sq]), newP = pieceOf(after[sq]);
    if (!oldP || !newP || oldP.player !== newP.player || oldP.type !== newP.type) continue;
    rotated.push(sq);
    gone.splice(i, 1);
    came.splice(j, 1);
  }
  if (rotated.length && !gone.length && !came.length) {
    // Pure rotation move -- animate the spin and return.
    for (const sq of rotated) {
      const node = svg.querySelector(`[data-piece="${sq}"]`);
      if (!node) continue;
      const oldP = pieceOf(before[sq]), newP = pieceOf(after[sq]);
      const headings = (SESSION.ruleset.facingAngles || [])[newP.player] || [];
      const oldAngle = headings[oldP.facing || 0] || 0;
      let newAngle = headings[newP.facing || 0] || 0;
      // Pick the shortest arc: if old=270 and new=0, spin +90 not -270.
      let delta = newAngle - oldAngle;
      if (delta > 180) delta -= 360;
      if (delta < -180) delta += 360;
      const [bx, by] = squareXY(sq, cell, view);
      const cx = bx + cell / 2, cy = by + cell / 2;
      const img = node.querySelector('image');
      if (img) {
        img.animate([
          { transform: `rotate(${oldAngle}deg)`, transformOrigin: `${cx}px ${cy}px` },
          { transform: `rotate(${oldAngle + delta}deg)`, transformOrigin: `${cx}px ${cy}px` }
        ], { duration: PACE.rotate, easing: 'ease-in-out' });
      }
      busy = Math.max(busy, PACE.rotate);
      landed = Math.max(landed, PACE.rotate);
    }
    return nothing();
  }

  const at = sq => squareXY(sq, cell, view);
  // The step a move takes is the mover's to make, so which seams are open is asked of
  // the player whose piece arrived. It matters on a one-way board: the rook that may
  // not cross the left edge slid the long way round the board, and drawing it through
  // the seam would show a move that side is not allowed.
  const stepOf = (from, to) => wrapStep(from, to, W, H,
    wrapAllow(SESSION.ruleset.wrap, (pieceOf(after[to]) || {}).player));
  const pairs = [];

  // The move the player actually made comes first, so a piece that landed where another
  // just left is matched to its own origin rather than to whatever happens to fit.
  if (last && last.from !== null && last.from !== undefined
      && gone.includes(last.from) && came.includes(last.to)) {
    pairs.push([last.from, last.to]);
  }
  // Then anything else that moved without changing what it is: the castling rook.
  for (const to of came) {
    if (pairs.some(p => p[1] === to)) continue;
    const from = gone.find(sq =>
      !pairs.some(p => p[0] === sq) && before[sq] === after[to]);
    if (from !== undefined) pairs.push([from, to]);
  }

  const moved = new Set(pairs.map(p => p[0]));
  // When each travelling piece lands, and which way it came from. Both are for the
  // capture below, which has to happen *to* the victim rather than beside it.
  const blows = new Map();
  let lastBlow = 0;
  // A piece that came through a portal, kept to be laid over the piece it takes (Q836).
  let traveller = null;
  for (const [from, to] of pairs) {
    let node = swallowed.has(to) ? null : svg.querySelector(`[data-piece="${to}"]`);
    const [fx, fy] = at(from), [tx, ty] = at(to);
    if (!node) {
      if (!swallowed.has(to)) continue;
      const piece = pieceOf(after[to]);
      if (!piece) continue;
      node = ghostAt(svg, piece, tx, ty, cell);
      doomed.set(to, node);
    }
    svg.appendChild(node);            // travelling pieces pass over stationary ones
    // Schema 19: a move through a portal is one journey (Q836), the engine's two squares
    // for it read off the move (`portal`, `portalOut`): the piece slides to the end it
    // goes in by, shrinks and spins down into it, grows and spins up out of the other end,
    // and settles one step on, where it lands, and takes there. Each leg along the
    // mover's own step.
    // Schema 22 (Q843d): a move through a fold is one journey too, a ripple in place of the
    // spin: onto the fold, down into the pool, and up out of the far end, where it ends
    // (`foldTrip`, schema 24).
    if (last && last.foldIn !== undefined && last.foldOut !== undefined
        && from === last.from && to === last.to && !swallowed.has(to)) {
      const time = foldTrip(svg, node, pieceOf(before[from]), before, last, cell, W, H, view);
      traveller = node;
      blows.set(to, time.blow);
      lastBlow = Math.max(lastBlow, time.blow.at);
      landed = Math.max(landed, time.landed);
      busy = Math.max(busy, time.landed);
      continue;
    }
    if (last && last.portal !== undefined && last.portalOut !== undefined
        && from === last.from && to === last.to && !swallowed.has(to)) {
      const time = portalTrip(svg, node, pieceOf(before[from]), before, last, cell, W, H,
                              view);
      traveller = node;
      blows.set(to, time.blow);
      lastBlow = Math.max(lastBlow, time.blow.at);
      landed = Math.max(landed, time.landed);
      busy = Math.max(busy, time.landed);
      continue;
    }
    const step = stepOf(from, to);
    const arc = pathHops(before, from, step, W, H, view);
    // Where the piece came from, along the step rather than across the board. The two
    // agree on every square pair that does not cross a seam, which is every move of
    // every board with edges.
    const [dx, dy] = stepXY(step, cell, view);
    const flight = travel(node, -dx, -dy, arc);
    if (step.wrapped) {
      animateCrossing(svg, node, pieceOf(before[from]), fx, fy, dx, dy, cell, arc, flight);
    }
    // Struck from the direction the piece actually arrived from, which through a seam
    // is the opposite of the way the two squares sit on screen.
    blows.set(to, { at: flight * PACE.impact, dx: dx, dy: dy });
    lastBlow = Math.max(lastBlow, flight * PACE.impact);
    landed = Math.max(landed, flight);
    busy = Math.max(busy, flight);
  }

  // Whatever left a square without arriving anywhere: a capture, an en passant victim,
  // a piece caught in an explosion. It is not in the new position, so it is redrawn as a
  // ghost purely to be taken off it -- and the taking waits for whatever killed it to get
  // there. It used to start at once and take less time than the move did, so on anything
  // longer than a step the victim was gone before the capturing piece arrived and a
  // capture read as a piece quietly giving up. A victim with no attacker of its own -- an
  // en passant pawn, a bystander in an explosion -- is struck when the last travelling
  // piece lands, because that is the move that killed it.
  //
  // When the ruleset has an explosion rule, victims inside the blast radius of any
  // capture destination get the fiery explosion animation rather than the plain
  // knock-off, and immune pieces that survived the blast get a brief shield shimmer.
  const ex = SESSION.ruleset.explosion;
  const blastVictims = new Set();
  // Schema 10 (Q18): `by` and `of` narrow *which* captures explode at all, and the page
  // cannot work that out from two positions -- a capture by a piece that is not a bomb
  // looks exactly like one by a piece that is. Null means every capturer / every victim,
  // which is atomic and every ruleset written before schema 10.
  const explodes = (from, to) => {
    if (!ex) return false;
    const victim = pieceOf(before[to]);
    if (!victim) return false;                 // nothing was taken here
    if (ex.of && !ex.of[victim.type]) return false;
    if (ex.by) {
      if (from === null || from === undefined) return false;
      const capturer = pieceOf(before[from]);
      if (!capturer || !ex.by[capturer.type]) return false;
    }
    return true;
  };
  if (ex) {
    // Paired moves that landed on a square with an explosion neighbourhood.
    for (const [from, to] of pairs) {
      if (!explodes(from, to)) continue;
      for (const neighbour of (ex.blast[to] || [])) {
        if (gone.includes(neighbour) && !moved.has(neighbour)) blastVictims.add(neighbour);
      }
    }
    // Atomic capture where the capturer also dies: from→to never pairs because
    // `to` is not in `came` (nothing survives there). The capturer still has to
    // *travel* to `to` before the blast fires, so a ghost is animated along the
    // move path and then exploded at the destination. The victim at `to` and
    // every blast neighbour all get the explosion timed to the capturer's arrival.
    if (last && last.from !== null && last.from !== undefined
        && gone.includes(last.from) && gone.includes(last.to)
        && !came.includes(last.to) && explodes(last.from, last.to)) {
      // Animate the capturer travelling from `from` to `to` as a ghost.
      const capturerPiece = pieceOf(before[last.from]);
      if (capturerPiece) {
        const [fx, fy] = at(last.from), [tx, ty] = at(last.to);
        // Nothing of the capturer survives, so the mover is read off the piece that
        // left rather than off a piece that arrived.
        const step = wrapStep(last.from, last.to, W, H,
          wrapAllow(SESSION.ruleset.wrap, capturerPiece.player));
        const arc = pathHops(before, last.from, step, W, H, view);
        const [dx, dy] = stepXY(step, cell, view);
        const ghost = document.createElementNS(SVGNS, 'g');
        ghost.innerHTML = drawPiece(capturerPiece, tx, ty, cell);
        ghost.setAttribute('pointer-events', 'none');
        svg.appendChild(ghost);
        const flight = travel(ghost, -dx, -dy, arc);
        if (step.wrapped) {
          animateCrossing(svg, ghost, capturerPiece, fx, fy, dx, dy, cell, arc, flight);
        }
        const arriveAt = flight * PACE.impact;
        // The capturer explodes at `to` on arrival, not at `from`.
        const capBlow = { at: arriveAt, dx: dx, dy: dy };
        busy = Math.max(busy,
          explode(svg, capturerPiece, tx, ty, cell, capBlow));
        ghost.animate([{ opacity: 1 }, { opacity: 0 }],
          { duration: PACE.explosion, delay: arriveAt, fill: 'forwards' });
        // All other blast victims are timed to this arrival.
        lastBlow = Math.max(lastBlow, arriveAt);
        busy = Math.max(busy, flight);
      }
      blastVictims.add(last.to);    // the victim
      // `last.from` is NOT added: the capturer was animated above at `to`.
      moved.add(last.from);         // skip the ghost-at-origin loop below
      for (const neighbour of (ex.blast[last.to] || [])) {
        if (gone.includes(neighbour) && !moved.has(neighbour)) blastVictims.add(neighbour);
      }
    }
  }

  for (const sq of gone) {
    if (moved.has(sq)) continue;
    const piece = pieceOf(before[sq]);
    if (!piece) continue;
    const [x, y] = at(sq);
    if (blastVictims.has(sq)) {
      busy = Math.max(busy,
        explode(svg, piece, x, y, cell, blows.get(sq) || { at: lastBlow, dx: 0, dy: 0 }));
    } else {
      busy = Math.max(busy,
        capture(svg, piece, x, y, cell, blows.get(sq) || { at: lastBlow, dx: 0, dy: 0 }));
    }
  }
  // The piece that came out of a portal is laid over the one it takes, which is drawn
  // after it: it grows up out of the end and lands on its victim, and a victim knocked
  // off on top of it would hide the arrival the whole journey was leading to.
  if (traveller) svg.appendChild(traveller);

  // Immune pieces that were inside the blast radius but survived: a brief shield
  // shimmer tells the player the piece was in danger and resisted.  Only fires when
  // an explosion actually occurred -- not on quiet moves or regular captures.
  if (ex && blastVictims.size > 0) {
    // Blast centers are only destinations where a capture triggered an explosion,
    // not every paired move destination (which includes quiet moves).
    const blastCenters = new Set();
    for (const [from, to] of pairs) {
      if (gone.includes(to) && explodes(from, to)) blastCenters.add(to);
    }
    if (last && last.from !== null && last.from !== undefined
        && gone.includes(last.to) && !came.includes(last.to)
        && explodes(last.from, last.to)) {
      blastCenters.add(last.to);
    }
    // The shield shimmer fires at the same time as the explosion, so the player
    // reads "blast + survived" as one event rather than two separate beats.
    const resistDelay = lastBlow;
    const arrivedAt = new Set(pairs.map(p => p[1]));
    const shimmered = new Set();
    for (const center of blastCenters) {
      for (const neighbour of (ex.blast[center] || [])) {
        if (shimmered.has(neighbour)) continue;
        if (blastVictims.has(neighbour) || moved.has(neighbour)) continue;
        // Don't shimmer a square where a piece just arrived via travel animation.
        if (arrivedAt.has(neighbour)) continue;
        if (!after[neighbour]) continue;
        const nPiece = pieceOf(after[neighbour]);
        if (!nPiece) continue;
        if (!ex.immune[nPiece.type]) continue;
        shimmered.add(neighbour);
        const [nx, ny] = at(neighbour);
        busy = Math.max(busy, resist(svg, nx, ny, cell, resistDelay));
      }
    }
  }

  // And whatever arrived from nowhere: a drop.
  for (const sq of came) {
    if (pairs.some(p => p[1] === sq)) continue;
    if (swallowed.has(sq)) continue;
    const node = svg.querySelector(`[data-piece="${sq}"]`);
    if (!node) continue;
    node.animate([{ opacity: 0 }, { opacity: 1 }], { duration: PACE.drop, easing: 'ease-out' });
    landed = Math.max(landed, PACE.drop);
    busy = Math.max(busy, PACE.drop);
  }
  return nothing();
}

// A trip through a portal (Q836, `docs/archive/plan-portals.md` *Travel through a portal,
// animated*), as times in milliseconds from the start of the move: the slide to the end
// it goes in by (`approach`), the shrink and spin down into it (`dive`), which starts
// before the slide is quite over so the two read as one movement, the instant it is
// nowhere, the grow and spin up out of the other end and the step on to where it lands
// (`emerge`), the blow a piece standing there is struck with (`strike`), and when it is
// at rest (`landed`). `distance` is the slide's length in viewBox units. Pure, so the
// order of the steps is something a test can hold.
function portalJourney(distance) {
  const approach = Math.min(PACE.portalApproachMax,
                            PACE.portalApproachMin + distance * PACE.portalApproachPerUnit);
  const dive = { at: approach * PACE.portalOverlap, for: PACE.portalDive };
  const emerge = { at: dive.at + dive.for + PACE.portalVoid, for: PACE.portalEmerge };
  return {
    approach: { at: 0, for: approach }, dive: dive, emerge: emerge,
    strike: emerge.at + emerge.for * PACE.portalImpact,
    landed: emerge.at + emerge.for,
  };
}

// The slide, for the piece going in and, on a slide across the seam, for the copy of it
// leaving by the near edge: both the same motion, one repeat apart. Eased in, with a hair
// of give at the start, so it gathers speed into the portal as though pulled, rather
// than coasting to a stop beside it; a leap lifts over what it passes, as a travel does.
function portalSlide(el, dx, dy, arc, duration) {
  const frames = [{ transform: `translate(${dx}px, ${dy}px)` }];
  if (arc) {
    const lift = Math.min(Math.hypot(dx, dy) * 0.45, 6);
    frames.push({ transform: `translate(${dx / 2}px, ${dy / 2 - lift}px) scale(1.1)`,
                  offset: 0.5 });
  }
  frames.push({ transform: 'translate(0px, 0px)' });
  el.animate(frames, { duration: duration, fill: 'forwards',
                       easing: arc ? 'cubic-bezier(.45,0,.7,.6)' : 'cubic-bezier(.5,-.12,.75,.5)' });
}

// The way a piece went into a portal, as the engine says it went: the shortest step to
// the end is the way in on a board with edges, but round a seam the piece may have come
// the long way, and it comes out going the way it went in (`exitDir`, the sign of the
// move's own vector, `engine/compile.py`'s `PORTAL_DIRECTIONS` order). An axis whose
// shortest step runs against that sign is taken the other way round the board. Nothing
// is worked out but which of the step's two drawings to show; on a hexagon, whose seams
// are sides, the step stands.
const PORTAL_SIGNS = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
function portalWayIn(step, exitDir, W, H) {
  const sign = PORTAL_SIGNS[exitDir];
  if (!sign || step.files === undefined) return step;
  const round = (s, want, size) => (want && s && Math.sign(s) !== want ? s + want * size : s);
  const files = round(step.files, sign[0], W), ranks = round(step.ranks, sign[1], H);
  if (files === step.files && ranks === step.ranks) return step;
  return { files: files, ranks: ranks, wrapped: true };
}

// What the swirl over the portal end standing on a square of `board` is drawn as: `ink`,
// the colour it falls back to, the pair's own (`--pair-N`, Q838b: in at yellow, out at
// yellow), and `mark`, `pair` and `end`, the same names the glyph carries (`data-portal`),
// for a stylesheet to key a look on. The one place the trip asks what an end looks like,
// so a new glyph changes this and no timing.
function portalLook(board, square) {
  const end = portalEnd((pieceOf(board[square]) || {}).type);
  return {
    ink: end ? `var(--pair-${end.pair})` : 'var(--portal-edge)',
    mark: end ? `${end.pair}${end.end[0]}` : '', pair: end ? end.pair : '',
    end: end ? end.end : '',
  };
}

// A dashed ring in its pair's colour, edged so it reads on both woods, turning the way the
// piece turns: drawn in towards the centre as the piece goes down (`inward`), thrown out
// from it as the piece comes up. Laid under `under` when there is one, so the piece is
// seen inside it. Gone when it is done; invisible before it starts.
//
// Its look is a class and two custom properties, never a colour of its own: the ring is
// `.portal-swirl` (with `data-swirl` in or out and the end's `data-portal`, `data-pair`
// and `data-end`), stroked in `--swirl-ink` and edged in `--swirl-edge`, each falling back
// to `portalLook`'s colour and `--portal-edge`. A stylesheet rule that sets either
// property restyles the ring and leaves the motion alone.
function portalSwirl(svg, cx, cy, cell, look, inward, spin, at, duration, under) {
  const ring = document.createElementNS(SVGNS, 'g');
  ring.setAttribute('data-journey', inward ? 'swirl-in' : 'swirl-out');
  ring.setAttribute('class', 'portal-swirl');
  ring.setAttribute('data-swirl', inward ? 'in' : 'out');
  ring.setAttribute('data-portal', look.mark);
  ring.setAttribute('data-pair', look.pair);
  ring.setAttribute('data-end', look.end);
  ring.setAttribute('pointer-events', 'none');
  const dash = `stroke-dasharray="${cell * 0.17} ${cell * 0.13}" stroke-linecap="round"`;
  ring.innerHTML =
    `<circle cx="${cx}" cy="${cy}" r="${cell * 0.4}" fill="none"
      stroke="var(--swirl-edge, var(--portal-edge))" stroke-width="${cell * 0.1}" ${dash}/>`
    + `<circle cx="${cx}" cy="${cy}" r="${cell * 0.4}" fill="none"
      stroke="var(--swirl-ink, ${look.ink})" stroke-width="${cell * 0.055}" ${dash}/>`;
  ring.style.opacity = '0';
  ring.style.transformOrigin = `${cx}px ${cy}px`;
  if (under && under.parentNode === svg) svg.insertBefore(ring, under);
  else svg.appendChild(ring);
  const turn = deg => `rotate(${spin * deg}deg)`;
  const frames = inward
    ? [{ opacity: 0, transform: `${turn(0)} scale(1.45)` },
       { opacity: 0.95, transform: `${turn(110)} scale(1)`, offset: 0.35 },
       { opacity: 0, transform: `${turn(320)} scale(0.15)` }]
    : [{ opacity: 0, transform: `${turn(0)} scale(0.2)` },
       { opacity: 0.95, transform: `${turn(120)} scale(1)`, offset: 0.3 },
       { opacity: 0, transform: `${turn(300)} scale(1.55)` }];
  const run = ring.animate(frames, { duration: duration, delay: at, fill: 'forwards',
                                     easing: inward ? 'cubic-bezier(.4,0,.9,.5)'
                                                    : 'cubic-bezier(.1,.5,.3,1)' });
  run.onfinish = () => ring.remove();
  return at + duration;
}

// The end itself reacting: a gulp as the piece goes down into it, a kick as one comes up
// out of it. The glyph the board drew, scaled about its own centre and given back
// untouched (no fill), so the end at rest is the end the drawing made.
function portalPulse(svg, square, cell, view, frames, at, duration) {
  const glyph = svg.querySelector(`[data-piece="${square}"]`);
  if (!glyph || typeof glyph.animate !== 'function') return 0;
  const [x, y] = squareXY(square, cell, view);
  glyph.style.transformOrigin = `${x + cell / 2}px ${y + cell / 2}px`;
  const run = glyph.animate(frames, { duration: duration, delay: at, easing: 'ease-in-out' });
  run.onfinish = () => { glyph.style.transformOrigin = ''; };
  return at + duration;
}

// A move through a portal as one journey (Q836). A ghost of the piece, drawn on the end it
// goes in by, slides there from where it stood, squashes against the rim, and shrinks and
// spins down into it while a swirl in the pair's colour closes over it and the end gulps.
// The piece the board already drew where it lands is held unseen at the other end until
// then, and comes up out of it spinning the same way, overshooting its size, hops the one
// step on, overshoots that and settles with a small squash, while that end kicks and a
// swirl opens round it. Every movement eases out past its mark and back (springy
// cubic-beziers, a keyframe of overshoot and squash), so it reads as a body thrown
// through a hole rather than a slide show. Returns when it lands and the blow a piece
// taken there is struck with: the moment the piece arrives on its square at full size.
function portalTrip(svg, node, piece, before, last, cell, W, H, view) {
  const at = sq => squareXY(sq, cell, view);
  // Both legs are the mover's, so its seams: the end it goes in by is nobody's piece.
  const allow = wrapAllow(SESSION.ruleset.wrap, last.player);
  const stepOf = (from, to) => wrapStep(from, to, W, H, allow);
  const [ex, ey] = at(last.portal), [fx, fy] = at(last.from), [tx, ty] = at(last.to);
  const into = portalWayIn(stepOf(last.from, last.portal), last.exitDir, W, H);
  const arc = pathHops(before, last.from, into, W, H, view);
  const [idx, idy] = stepXY(into, cell, view);
  const out = stepOf(last.portalOut, last.to);
  const [odx, ody] = stepXY(out, cell, view);
  const plan = portalJourney(Math.hypot(idx, idy));
  // Which way it turns: clockwise going right (or down), as a wheel rolling that way
  // would, and the same way coming out as going in, so the two halves are one spin.
  const spin = (Math.abs(idx) >= Math.abs(idy) ? idx : idy) < 0 ? -1 : 1;
  const turn = deg => `rotate(${spin * deg}deg)`;
  const ecx = ex + cell / 2, ecy = ey + cell / 2;

  if (piece) {
    const goer = ghostAt(svg, piece, ex, ey, cell);
    goer.setAttribute('data-journey', 'approach');
    portalSlide(goer, -idx, -idy, arc, plan.approach.for);
    if (into.wrapped) {
      // Across the seam: the copy leaving by the near edge makes the same slide, and the
      // two hand over in the middle, as every crossing does (`animateCrossing`).
      goer.animate(crossOpacity(false), { duration: plan.approach.for, fill: 'backwards' });
      const away = ghostAt(svg, piece, fx + idx, fy + idy, cell);
      portalSlide(away, -idx, -idy, arc, plan.approach.for);
      const fade = away.animate(crossOpacity(true),
                                { duration: plan.approach.for, fill: 'both' });
      fade.onfinish = () => away.remove();
    }
    // The dive, on the drawing inside the ghost, about the end's centre: a squash against
    // the rim and a turn back to wind up, then down and round, gone at the end.
    const inner = goer.firstElementChild || goer;
    if (inner !== goer) inner.setAttribute('data-journey', 'dive');
    inner.style.transformOrigin = `${ecx}px ${ecy}px`;
    const dive = inner.animate([
      { opacity: 1, transform: `${turn(0)} scale(1, 1)`, easing: 'cubic-bezier(.3,.6,.5,1)' },
      { opacity: 1, transform: `${turn(-14)} scale(1.14, 0.86)`, offset: 0.22,
        easing: 'cubic-bezier(.5,0,.85,.45)' },
      { opacity: 1, transform: `${turn(150)} scale(0.55)`, offset: 0.6,
        easing: 'cubic-bezier(.5,0,1,.6)' },
      { opacity: 0, transform: `${turn(400)} scale(0.02)` },
    ], { duration: plan.dive.for, delay: plan.dive.at, fill: 'forwards' });
    dive.onfinish = () => goer.remove();
    portalSwirl(svg, ecx, ecy, cell, portalLook(before, last.portal), true, spin,
                plan.dive.at, plan.dive.for + 60, goer);
  }
  portalPulse(svg, last.portal, cell, view, [
    { transform: 'scale(1)' },
    { transform: 'scale(1.12)', offset: 0.3 },
    { transform: 'scale(0.84)', offset: 0.62 },
    { transform: 'scale(1.05)', offset: 0.84 },
    { transform: 'scale(1)' },
  ], plan.dive.at, plan.dive.for + 120);

  // Out of the other end. Where it comes up is the square one step back from where it
  // lands along its own step out, which is the exit end, or that end's copy across the
  // seam when the step out crosses it, so the swirl and the piece meet there.
  const ocx = tx - odx + cell / 2, ocy = ty - ody + cell / 2;
  portalPulse(svg, last.portalOut, cell, view, [
    { transform: 'scale(1)' },
    { transform: 'scale(0.86)', offset: 0.22 },
    { transform: 'scale(1.16)', offset: 0.5 },
    { transform: 'scale(0.97)', offset: 0.78 },
    { transform: 'scale(1)' },
  ], Math.max(0, plan.emerge.at - 40), 340);
  portalSwirl(svg, ocx, ocy, cell, portalLook(before, last.portalOut), false, spin,
              Math.max(0, plan.emerge.at - 30), 380, node);
  // The piece itself, about its own centre where it lands: held small and unseen at the
  // exit until its moment (`backwards`), popped up past full size while it is still over
  // the end, thrown the step on past its square with a squash across the way it moves,
  // and settled. No fill after: at rest it is exactly what the board drew.
  const across = Math.abs(odx) >= Math.abs(ody);
  const squash = across ? 'scale(0.9, 1.08)' : 'scale(1.08, 0.9)';
  const give = across ? 'scale(1.03, 0.97)' : 'scale(0.97, 1.03)';
  const shift = k => `translate(${-odx * k}px, ${-ody * k}px)`;
  node.style.transformOrigin = `${tx + cell / 2}px ${ty + cell / 2}px`;
  const emerge = node.animate([
    { opacity: 0, transform: `${shift(1)} ${turn(-320)} scale(0.04)`,
      easing: 'cubic-bezier(.2,.75,.35,1)' },
    { opacity: 1, transform: `${shift(0.9)} ${turn(-40)} scale(1.2)`, offset: 0.4,
      easing: 'cubic-bezier(.45,0,.35,1)' },
    { opacity: 1, transform: `${shift(-0.1)} ${turn(8)} ${squash}`, offset: 0.68,
      easing: 'cubic-bezier(.3,0,.4,1)' },
    { opacity: 1, transform: `${shift(0.02)} ${turn(-2)} ${give}`, offset: 0.86,
      easing: 'ease-out' },
    { opacity: 1, transform: `${shift(0)} ${turn(0)} scale(1, 1)` },
  ], { duration: plan.emerge.for, delay: plan.emerge.at, fill: 'backwards' });
  emerge.onfinish = () => { node.style.transformOrigin = ''; };
  return { landed: plan.landed, blow: { at: plan.strike, dx: odx, dy: ody } };
}

// A trip through a fold (schema 24, `docs/archive/plan-portals-2.md` §5.3R), as times in
// milliseconds from the start of the move: the slide onto the fold it goes in by
// (`approach`), the sink into the pool (`sink`), starting before the slide is quite over,
// the instant under the surface, and the rise out of the far end (`rise`), where the move
// ends: nothing slides on. A piece standing there is struck once the rise is done (`strike`),
// the ordinary capture beat, and the piece is at rest then too (`landed`). `inDistance` is
// the slide's length in viewBox units. Pure, as `portalJourney` is, so the order of the steps
// is something a test can hold.
function foldJourney(inDistance) {
  const approach = Math.min(PACE.portalApproachMax,
                            PACE.portalApproachMin + inDistance * PACE.portalApproachPerUnit);
  const sink = { at: approach * PACE.portalOverlap, for: PACE.foldSink };
  const rise = { at: sink.at + sink.for + PACE.foldUnder, for: PACE.foldRise };
  const landed = rise.at + rise.for;
  return { approach: { at: 0, for: approach }, sink: sink, rise: rise, strike: landed, landed: landed };
}

// A ring of ripple on a pool, in its group's colour cased in the edge: spreading out from
// the middle and fading as a piece sinks in (`inward`) or rises out. Laid under `under` when
// there is one, so the piece is seen inside it. Gone when it is done; unseen before it starts.
function foldRipple(svg, cx, cy, cell, pair, inward, at, under) {
  const ring = document.createElementNS(SVGNS, 'g');
  ring.setAttribute('data-journey', inward ? 'ripple-in' : 'ripple-out');
  ring.setAttribute('class', 'fold-ripple');
  ring.setAttribute('data-pair', String(pair));
  ring.setAttribute('pointer-events', 'none');
  const circle = (r, stroke, width) => `<circle cx="${cx}" cy="${cy}" r="${cell * r}" fill="none"
    stroke="${stroke}" stroke-width="${cell * width}"/>`;
  ring.innerHTML = circle(0.2, 'var(--portal-edge)', 0.07) + circle(0.2, `var(--pair-${pair})`, 0.04)
    + circle(0.34, 'var(--portal-edge)', 0.06) + circle(0.34, `var(--pair-${pair})`, 0.03);
  ring.style.opacity = '0';
  ring.style.transformOrigin = `${cx}px ${cy}px`;
  if (under && under.parentNode === svg) svg.insertBefore(ring, under);
  else svg.appendChild(ring);
  const run = ring.animate([
    { opacity: 0.95, transform: 'scale(0.5)' },
    { opacity: 0.7, transform: 'scale(1.05)', offset: 0.45 },
    { opacity: 0, transform: 'scale(1.5)' },
  ], { duration: PACE.foldRipple, delay: at, fill: 'forwards', easing: 'cubic-bezier(.2,.6,.35,1)' });
  run.onfinish = () => ring.remove();
  return at + PACE.foldRipple;
}

// The pool itself answering: a dip as the piece sinks in, a swell as one rises out. The pool
// the board drew (`data-fold-at`, its outer group, as the vortex's pulse is), scaled about its
// own centre and given back untouched.
function foldPulse(svg, square, cell, view, frames, at, duration) {
  const pool = svg.querySelector(`[data-fold-at="${square}"]`);
  if (!pool || typeof pool.animate !== 'function') return 0;
  const [x, y] = squareXY(square, cell, view);
  pool.style.transformOrigin = `${x + cell / 2}px ${y + cell / 2}px`;
  const run = pool.animate(frames, { duration: duration, delay: at, easing: 'ease-in-out' });
  run.onfinish = () => { pool.style.transformOrigin = ''; };
  return at + duration;
}

// A move through a fold as one journey (Q843d; Q867c, §5.3R). A ghost of the piece, drawn
// on the fold it goes in by, slides there from where it stood, dips against the surface and
// sinks into the pool, smaller and fading, with no turn, as a ring of ripple spreads and the
// pool dips. The piece the board already drew on the far end, where the move ends, is held
// unseen until then and rises out of it with the answering ripple and swell, and takes a
// piece standing there once it is up. The slide along the mover's own step. Returns when it
// lands and the blow, struck along the way the piece went in.
function foldTrip(svg, node, piece, before, last, cell, W, H, view) {
  const at = sq => squareXY(sq, cell, view);
  const allow = wrapAllow(SESSION.ruleset.wrap, last.player);
  const stepOf = (from, to) => wrapStep(from, to, W, H, allow);
  const [sx, sy] = at(last.foldIn), [tx, ty] = at(last.to);
  const into = stepOf(last.from, last.foldIn);
  const [idx, idy] = stepXY(into, cell, view);
  const plan = foldJourney(Math.hypot(idx, idy));
  const pair = last.foldPair || 1;
  const scx = sx + cell / 2, scy = sy + cell / 2;

  if (piece) {
    const goer = ghostAt(svg, piece, sx, sy, cell);
    goer.setAttribute('data-journey', 'approach');
    portalSlide(goer, -idx, -idy, pathHops(before, last.from, into, W, H, view), plan.approach.for);
    // The sink, on the drawing inside the ghost, about the fold's centre: a dip against the
    // surface, then down and away, no turn.
    const inner = goer.firstElementChild || goer;
    if (inner !== goer) inner.setAttribute('data-journey', 'sink');
    inner.style.transformOrigin = `${scx}px ${scy}px`;
    const sink = inner.animate([
      { opacity: 1, transform: 'translate(0px, 0px) scale(1, 1)', easing: 'cubic-bezier(.3,.6,.5,1)' },
      { opacity: 1, transform: `translate(0px, ${cell * 0.04}px) scale(1.08, 0.9)`, offset: 0.25,
        easing: 'cubic-bezier(.5,0,.85,.5)' },
      { opacity: 0.55, transform: `translate(0px, ${cell * 0.08}px) scale(0.7)`, offset: 0.65,
        easing: 'cubic-bezier(.5,0,1,.6)' },
      { opacity: 0, transform: `translate(0px, ${cell * 0.1}px) scale(0.4)` },
    ], { duration: plan.sink.for, delay: plan.sink.at, fill: 'forwards' });
    sink.onfinish = () => goer.remove();
    foldRipple(svg, scx, scy, cell, pair, true, plan.sink.at, goer);
  }
  foldPulse(svg, last.foldIn, cell, view, [
    { transform: 'scale(1)' }, { transform: 'scale(0.9)', offset: 0.4 },
    { transform: 'scale(1.04)', offset: 0.75 }, { transform: 'scale(1)' },
  ], plan.sink.at, plan.sink.for + 120);

  // Out of the far end, where the move ends: the pool swells and the ripple answers there.
  const ocx = tx + cell / 2, ocy = ty + cell / 2;
  foldPulse(svg, last.foldOut, cell, view, [
    { transform: 'scale(1)' }, { transform: 'scale(1.1)', offset: 0.4 },
    { transform: 'scale(0.97)', offset: 0.75 }, { transform: 'scale(1)' },
  ], Math.max(0, plan.rise.at - 30), 340);
  foldRipple(svg, ocx, ocy, cell, pair, false, Math.max(0, plan.rise.at - 20), node);
  // The piece itself, about its own centre: held small and unseen on the far end until its
  // moment (`backwards`), rising past full size and settling. No fill after: at rest it is
  // exactly what the board drew.
  node.style.transformOrigin = `${ocx}px ${ocy}px`;
  const rise = node.animate([
    { opacity: 0, transform: 'scale(0.4)', easing: 'cubic-bezier(.2,.75,.35,1)' },
    { opacity: 1, transform: 'scale(1.08)', offset: 0.7, easing: 'ease-out' },
    { opacity: 1, transform: 'scale(1)' },
  ], { duration: plan.rise.for, delay: plan.rise.at, fill: 'backwards' });
  rise.onfinish = () => { node.style.transformOrigin = ''; };
  return { landed: plan.landed, blow: { at: plan.strike, dx: idx, dy: idy } };
}

// With reduced motion, a trip through a portal is a crossfade and nothing else: the
// piece fades from the square it left while it fades in on the square it landed on, and
// a piece taken there fades as it arrives. No slide, no turn, no change of size. Any
// other move draws at once, as it always has with motion reduced; this returns 0 for it.
function portalStill(svg, before, after, last, cell, view) {
  // Schema 22 (Q843d): a trip through a fold, too.
  if (!last || (last.portal === undefined && last.foldIn === undefined)
      || last.from === null || last.from === undefined
      || before[last.from] === after[last.from] || !after[last.to]) return 0;
  const node = svg.querySelector(`[data-piece="${last.to}"]`);
  if (!node) return 0;
  const half = PACE.portalStill * 0.55;
  const leaving = pieceOf(before[last.from]);
  if (leaving) {
    const [fx, fy] = squareXY(last.from, cell, view);
    const was = ghostAt(svg, leaving, fx, fy, cell);
    was.setAttribute('data-journey', 'still-from');
    was.animate([{ opacity: 1 }, { opacity: 0 }],
                { duration: half, easing: 'ease-out', fill: 'forwards' }).onfinish =
      () => was.remove();
  }
  const victim = before[last.to] && before[last.to] !== after[last.to]
    ? pieceOf(before[last.to]) : null;
  if (victim) {
    const [tx, ty] = squareXY(last.to, cell, view);
    const gone = ghostAt(svg, victim, tx, ty, cell);
    gone.setAttribute('data-journey', 'still-taken');
    gone.animate([{ opacity: 1 }, { opacity: 0 }],
                 { duration: half, delay: PACE.portalStill - half, easing: 'ease-in',
                   fill: 'forwards' }).onfinish = () => gone.remove();
  }
  svg.appendChild(node);
  node.animate([{ opacity: 0 }, { opacity: 1 }],
               { duration: half, delay: PACE.portalStill - half, easing: 'ease-in-out',
                 fill: 'backwards' });
  return PACE.portalStill;
}

// The second beat: the wild-piece phase, which the rules run after the mover's move
// inside the same `make`.
//
// The board runs it in that order too -- nothing here starts until the piece that moved
// has landed and stood still for `PACE.beat`. That is the fix for "a boar was captured
// by nothing": with both changes drawn as one transition there was no moment at which
// the boar was the thing moving, and the piece it ate went out beside a player's move
// that had not touched it.
//
// A wild piece is already drawn on the square it ends on, so its travel is the same
// backwards translation every other move gets, held at its origin until its beat begins,
// which is also what keeps it standing still while the mover moves in front of it.
function animatePhase(svg, phase, board, beat, cell, W, H, view) {
  const at = sq => squareXY(sq, cell, view);
  // A wild piece belongs to neither seat, so it crosses whatever the board's seams are
  // rather than one player's share of them.
  const allow = wrapAllow(SESSION.ruleset.wrap, null);
  const start = beat.landed + PACE.beat;
  let busy = 0;
  for (const step of phase) {
    const node = svg.querySelector(`[data-piece="${step.to}"]`);
    if (!node) continue;
    const [fx, fy] = at(step.from), [tx, ty] = at(step.to);
    svg.appendChild(node);
    // Generation 36 (N1): a taken wild piece coming back onto its start comes from nowhere
    // on the board, so it fades in where it stands, held invisible until its beat, and
    // strikes whatever it eats there as a step would.
    // Generation 41 (N3d): a copy grown by a splitting pouncer comes from nowhere on the
    // board too (its parent stays put), so it grows in the same way on the square it ate.
    if (step.kind === 'return' || step.kind === 'copy') {
      const appear = PACE.travelMin;
      node.animate([{ opacity: 0 }, { opacity: 1 }],
                   { duration: appear, delay: start, fill: 'backwards', easing: 'ease-out' });
      busy = Math.max(busy, start + appear);
      const eaten = step.eaten ? pieceOf(step.eaten) : null;
      if (eaten) {
        const ghost = beat.doomed.get(step.to) || ghostAt(svg, eaten, tx, ty, cell);
        busy = Math.max(busy, strike(svg, ghost, tx, ty, cell,
                                     { at: start + appear * PACE.impact, dx: 0, dy: 0 }));
      }
      continue;
    }
    // Generation 38 (N5): a turning-back piece blocked both ways turns where it stands.
    if (step.from === step.to) continue;
    const walk = wrapStep(step.from, step.to, W, H, allow);
    const arc = pathHops(board, step.from, walk, W, H, view);
    const [dx, dy] = stepXY(walk, cell, view);
    const flight = travel(node, -dx, -dy, arc, start);
    if (walk.wrapped) {
      animateCrossing(svg, node, pieceOf(board[step.to]), fx, fy, dx, dy, cell, arc,
                      flight, start);
    }
    busy = Math.max(busy, start + flight);
    if (!step.eaten) continue;
    const piece = pieceOf(step.eaten);
    if (!piece) continue;
    // Struck by the wild piece, from the direction it came: the same blow a player's
    // capture gets, so a viewer reads the two as the same kind of event.
    const blow = { at: start + flight * PACE.impact, dx: dx, dy: dy };
    const ghost = beat.doomed.get(step.to) || ghostAt(svg, piece, tx, ty, cell);
    busy = Math.max(busy, strike(svg, ghost, tx, ty, cell, blow));
  }
  return busy;
}

// Returns how long the move takes, which is what a capture on the destination square
// has to wait for.
//
// `delay` is which beat of the ply this is. It is `backwards`-filled on purpose: a piece
// whose beat has not started yet is drawn where it *was*, not where it ends up, which is
// what lets a wild piece stand still on the square it came from while the player's move
// plays out in front of it.
function travel(node, dx, dy, arc, delay = 0) {
  const distance = Math.hypot(dx, dy);
  const duration = Math.min(PACE.travelMax, PACE.travelMin + distance * PACE.travelPerUnit);
  const frames = [{ transform: `translate(${dx}px, ${dy}px)` }];
  if (arc) {
    // A lift and a slight swell, so a leap reads as going over rather than through.
    const lift = Math.min(distance * 0.45, 6);
    frames.push({ transform: `translate(${dx / 2}px, ${dy / 2 - lift}px) scale(1.12)`, offset: 0.5 });
  }
  frames.push({ transform: 'translate(0px, 0px)' });
  node.animate(frames, { duration, delay, fill: delay ? 'backwards' : 'none',
                         easing: arc ? 'ease-in-out' : 'cubic-bezier(.3,.7,.3,1)' });
  return duration;
}

// A capture, in two parts, because either one alone is easy to miss: the victim is
// knocked off the square it was standing on, and a ring is struck where it stood. The
// ring is the part that reads at a glance on a crowded board, and it is the reason this
// is not simply a fade -- a piece disappearing is what *any* of these three facts looks
// like, and a capture is the one worth telling apart.
//
// `blow` says when the piece that killed this one arrives and which way it came from, so
// both halves are timed to the impact rather than to the redraw. Returns when the last of
// it finishes.
function capture(svg, piece, x, y, cell, blow) {
  return strike(svg, ghostAt(svg, piece, x, y, cell), x, y, cell, blow);
}

// A piece drawn on the board that the *position* no longer has: the victim of a capture,
// or a piece that moved onto a square something else has since taken it on. It stands
// there, at full opacity, until whatever happens to it happens -- an animation that has
// not started leaves its element showing its own style.
function ghostAt(svg, piece, x, y, cell) {
  const ghost = document.createElementNS(SVGNS, 'g');
  ghost.innerHTML = drawPiece(piece, x, y, cell);
  ghost.setAttribute('pointer-events', 'none');
  ghost.style.transformOrigin = `${x + cell / 2}px ${y + cell / 2}px`;
  svg.appendChild(ghost);
  return ghost;
}

// The taking itself, done to a ghost that is already on the board. Split out from
// `capture` because a piece can travel *first* and be taken where it landed -- the phase
// eating what the mover just moved into it -- and that has to be one piece the eye can
// follow rather than one that vanishes and another that appears.
function strike(svg, ghost, x, y, cell, blow) {
  const cx = x + cell / 2, cy = y + cell / 2;
  const reach = Math.hypot(blow.dx, blow.dy) || 1;
  const kx = blow.dx / reach * cell * 0.24, ky = blow.dy / reach * cell * 0.24;

  const fade = ghost.animate([
    { opacity: 0.95, transform: 'translate(0px, 0px) scale(1)' },
    { opacity: 0.8, transform: `translate(${kx * 0.55}px, ${ky * 0.55}px) scale(1.08)`,
      offset: 0.3 },
    { opacity: 0, transform: `translate(${kx}px, ${ky}px) scale(0.5)` },
  ], { duration: PACE.strike, delay: blow.at, easing: 'ease-in', fill: 'forwards' });
  fade.onfinish = () => ghost.remove();

  const ring = document.createElementNS(SVGNS, 'circle');
  ring.setAttribute('cx', cx);
  ring.setAttribute('cy', cy);
  ring.setAttribute('r', cell * 0.4);
  ring.setAttribute('fill', 'none');
  ring.setAttribute('stroke', 'var(--bad)');
  ring.setAttribute('stroke-width', cell * 0.07);
  ring.setAttribute('pointer-events', 'none');
  // Invisible until the blow lands: an animation that has not started yet leaves the
  // element showing its own style, and a ring sitting on the victim beforehand would
  // announce the capture before it happened.
  ring.style.opacity = '0';
  ring.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(ring);
  const flash = ring.animate([
    { opacity: 0.9, transform: 'scale(0.45)' },
    { opacity: 0, transform: 'scale(1.3)' },
  ], { duration: PACE.burst, delay: blow.at, easing: 'ease-out', fill: 'forwards' });
  flash.onfinish = () => ring.remove();

  return blow.at + Math.max(PACE.strike, PACE.burst);
}

// An explosion: visually distinct from a plain capture so the player reads "blast
// damage" rather than "someone took that piece". The victim is consumed in place --
// no directional knock, because nothing struck it from any particular direction --
// and the marker is a double ring in amber/orange rather than the single red one a
// capture uses, with an added filled flash at the centre for the fireball.
function explode(svg, piece, x, y, cell, blow) {
  const cx = x + cell / 2, cy = y + cell / 2;

  // The ghost piece burns away in place: a bright flash, then rapid shrink to nothing.
  const ghost = document.createElementNS(SVGNS, 'g');
  ghost.innerHTML = drawPiece(piece, x, y, cell);
  ghost.setAttribute('pointer-events', 'none');
  ghost.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(ghost);
  const burn = ghost.animate([
    { opacity: 1, transform: 'scale(1)', filter: 'brightness(1)' },
    { opacity: 0.9, transform: 'scale(1.15)', filter: 'brightness(1.8)', offset: 0.25 },
    { opacity: 0, transform: 'scale(0.3)', filter: 'brightness(2)' },
  ], { duration: PACE.explosion, delay: blow.at, easing: 'ease-in', fill: 'forwards' });
  burn.onfinish = () => ghost.remove();

  // Inner fireball: a filled circle that flares bright and shrinks.
  const fireball = document.createElementNS(SVGNS, 'circle');
  fireball.setAttribute('cx', cx);
  fireball.setAttribute('cy', cy);
  fireball.setAttribute('r', cell * 0.32);
  fireball.setAttribute('fill', 'var(--fx-fire)');
  fireball.setAttribute('pointer-events', 'none');
  fireball.style.opacity = '0';
  fireball.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(fireball);
  const flare = fireball.animate([
    { opacity: 0.85, transform: 'scale(0.5)' },
    { opacity: 0.5, transform: 'scale(1.1)', offset: 0.4 },
    { opacity: 0, transform: 'scale(1.4)' },
  ], { duration: PACE.explosion, delay: blow.at, easing: 'ease-out', fill: 'forwards' });
  flare.onfinish = () => fireball.remove();

  // Outer blast ring: expands wide and fades, amber-coloured.
  const ring1 = document.createElementNS(SVGNS, 'circle');
  ring1.setAttribute('cx', cx);
  ring1.setAttribute('cy', cy);
  ring1.setAttribute('r', cell * 0.35);
  ring1.setAttribute('fill', 'none');
  ring1.setAttribute('stroke', 'var(--fx-flare)');
  ring1.setAttribute('stroke-width', cell * 0.08);
  ring1.setAttribute('pointer-events', 'none');
  ring1.style.opacity = '0';
  ring1.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(ring1);
  const blast1 = ring1.animate([
    { opacity: 0.9, transform: 'scale(0.4)' },
    { opacity: 0, transform: 'scale(1.6)' },
  ], { duration: PACE.explosion, delay: blow.at, easing: 'ease-out', fill: 'forwards' });
  blast1.onfinish = () => ring1.remove();

  // Inner ring: a second, tighter ring for the double-ring look.
  const ring2 = document.createElementNS(SVGNS, 'circle');
  ring2.setAttribute('cx', cx);
  ring2.setAttribute('cy', cy);
  ring2.setAttribute('r', cell * 0.25);
  ring2.setAttribute('fill', 'none');
  ring2.setAttribute('stroke', 'var(--fx-spark)');
  ring2.setAttribute('stroke-width', cell * 0.06);
  ring2.setAttribute('pointer-events', 'none');
  ring2.style.opacity = '0';
  ring2.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(ring2);
  const blast2 = ring2.animate([
    { opacity: 0.8, transform: 'scale(0.5)' },
    { opacity: 0, transform: 'scale(1.35)' },
  ], { duration: PACE.explosion * 0.8, delay: blow.at, easing: 'ease-out', fill: 'forwards' });
  blast2.onfinish = () => ring2.remove();

  return blow.at + PACE.explosion;
}

// A resist shimmer: the piece was inside the blast radius but is immune to the
// explosion. A translucent shield briefly pulses around it, so the player sees
// "that piece was in danger and survived" rather than "nothing happened there".
function resist(svg, x, y, cell, delay) {
  const cx = x + cell / 2, cy = y + cell / 2;

  // A rounded-rect shield outline that pulses once and fades.
  const shield = document.createElementNS(SVGNS, 'rect');
  const pad = cell * 0.1;
  shield.setAttribute('x', x + pad);
  shield.setAttribute('y', y + pad);
  shield.setAttribute('width', cell - pad * 2);
  shield.setAttribute('height', cell - pad * 2);
  shield.setAttribute('rx', cell * 0.15);
  shield.setAttribute('ry', cell * 0.15);
  shield.setAttribute('fill', 'none');
  shield.setAttribute('stroke', 'var(--fx-shield)');
  shield.setAttribute('stroke-width', cell * 0.07);
  shield.setAttribute('pointer-events', 'none');
  shield.style.opacity = '0';
  shield.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(shield);
  const pulse = shield.animate([
    { opacity: 0, transform: 'scale(1.15)' },
    { opacity: 0.75, transform: 'scale(1)', offset: 0.3 },
    { opacity: 0.6, transform: 'scale(1.05)', offset: 0.6 },
    { opacity: 0, transform: 'scale(1.1)' },
  ], { duration: PACE.resist, delay, easing: 'ease-in-out', fill: 'forwards' });
  pulse.onfinish = () => shield.remove();

  // A faint filled overlay that flashes to reinforce the "shielded" read.
  const glow = document.createElementNS(SVGNS, 'rect');
  glow.setAttribute('x', x + pad);
  glow.setAttribute('y', y + pad);
  glow.setAttribute('width', cell - pad * 2);
  glow.setAttribute('height', cell - pad * 2);
  glow.setAttribute('rx', cell * 0.15);
  glow.setAttribute('ry', cell * 0.15);
  glow.setAttribute('fill', 'var(--fx-shield)');
  glow.setAttribute('pointer-events', 'none');
  glow.style.opacity = '0';
  glow.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(glow);
  const flash = glow.animate([
    { opacity: 0 },
    { opacity: 0.2, offset: 0.3 },
    { opacity: 0 },
  ], { duration: PACE.resist, delay, easing: 'ease-in-out', fill: 'forwards' });
  flash.onfinish = () => glow.remove();

  return delay + PACE.resist;
}

// ---------------------------------------------------------------- the side panel

function drawPane() {
  const pane = document.getElementById('pane');
  if (!pane || !SESSION) return;
  const game = DETAIL[SESSION.ruleset.hash];

  if (TAB === 'moves') {
    pane.className = 'tabpane moves';
    // Which beat the board is showing, while reviewing. The list is how a reader finds
    // the ply they want to be at, so it has to say which one they are at -- and it
    // follows the cursor rather than the end of the game, or stepping back ten moves
    // leaves the panel scrolled to a position nobody is looking at.
    const rows = moveRows();
    const want = AT ? (AT.kind === 'phase' ? 'phase' : 'move') : null;
    const here = AT ? rows.findIndex(r => (r.plies === AT.plies || r.also === AT.plies)
      && r.kind === want) : -1;
    pane.innerHTML = rows.map((r, i) => {
      const names = (r.stripe ? ['stripe'] : []).concat(r.kind === 'phase' ? ['phase'] : [],
                                                        r.kind === 'result' ? ['result'] : [],
                                                        r.took ? ['took'] : [],
                                                        i === here ? ['on'] : []);
      return `<div${names.length ? ` class="${names.join(' ')}"` : ''}>${r.html}</div>`;
    }).join('') || '<div class="muted small">No moves have been played yet.</div>';
    // A game between two people states its rules under the moves (test 7).
    if (SESSION.correspondence) pane.innerHTML += corrRulesHtml();
    const row = here >= 0 && pane.children ? pane.children[here] : null;
    if (row && row.scrollIntoView) row.scrollIntoView({ block: 'nearest' });
    else pane.scrollTop = pane.scrollHeight;
    return;
  }
  pane.className = 'tabpane';
  if (!game) { pane.innerHTML = '<div class="muted small">loading...</div>'; return; }

  if (TAB === 'rules') {
    pane.innerHTML = `<ul class="rules">${
      (game.ruleLines || []).map(r => `<li>${esc(r)}</li>`).join('')}</ul>
      <details class="u-mt-10">
        <summary class="muted small">where the pieces start</summary>
        <ul class="rules muted small u-mt-6">${
          (game.setupLines || []).map(r => `<li>${esc(r)}</li>`).join('')}</ul>
      </details>`;
    return;
  }

  pane.innerHTML = `<div class="pieces">${(game.pieces || []).map((piece, index) =>
    piecesHtml(piece, { index, selected: HIGHLIGHT === index,
                        ranks: pieceSpeeds(game.pieces) })).join('')}</div>
    <div class="muted small u-mt-8">Click a piece to find it on the board.</div>`;
  pane.querySelectorAll('[data-type]').forEach(node => {
    node.onclick = () => {
      const type = Number(node.dataset.type);
      HIGHLIGHT = HIGHLIGHT === type ? null : type;
      drawBoard();
      drawPane();
    };
  });
}

// The move list, one row per beat of the game. A ply is the move somebody made, and in a
// game with wild pieces also the phase the rules ran after it -- which the board animates
// and the review steps onto, so a list that said nothing about it was the one place a
// wolf's capture went unrecorded. The phase's row sits under its move and reads on from
// it: *1... Pe7-e6*, then *and the wolf on c6 steps to c5*.
//
// Everything in a row is what the core reported: the move's own text, and the phase's
// steps as `lastMove.phase` carried them, with the walkers `plyBeats` found when the
// timeline split the ply. A ply the timeline never split -- nothing on this page makes
// one today -- still gets its row, calling the walker a wild piece rather than guessing
// which. A ply with no phase is one row, exactly as it always was.
//
// The stripe is per ply, not per row, so a move and its phase read as one band. A phase
// row is quieter than a move's, since most of them are a patrol going round -- except one
// in which a wild piece took something, which is the line a reader is looking for.
function moveRows() {
  const split = new Map();
  if (TIMELINE_FOR === SESSION.id) {
    for (const entry of TIMELINE) if (entry.kind === 'phase') split.set(entry.plies, entry);
  }
  const rows = [];
  const outcome = SESSION.outcome;
  const last = SESSION.history.length - 1;
  // One move's row, and its wild pieces' walk under it.
  const plyRow = (m, i) => {
    const stripe = i % 2 === 0;
    rows.push({ plies: i + 1, kind: 'move', stripe: stripe, html: `${
      stripe ? `${Math.floor(i / 2) + 1}.` : '&nbsp;&nbsp;&hellip;'} ${esc(m.text)}${
      signHtml(m, i === last ? outcome : null)}${liftedHtml(m)}${lookedHtml(m)}${
      corrMovedDayHtml(i)}` });
    if (!(m.phase || []).length) return;
    const beat = split.get(i + 1);
    rows.push({ plies: i + 1, kind: 'phase', stripe: stripe,
                took: m.phase.some(step => step.eaten),
                html: esc(phaseText(m.phase, beat ? beat.walkers : null)) });
  };
  // Schema 26: a round is one row, both moves in seat order (`roundRowHtml`), its wild
  // pieces' walk under it, as a ply's is. A spy game's (schema 27) turns come first, a row
  // a move, and its rounds are named as rounds, so the two counts are not read as one.
  if (SESSION.round || SESSION.spy) {
    let i = 0;
    while (i < SESSION.history.length) {
      const first = SESSION.history[i], second = SESSION.history[i + 1];
      if (first.round === undefined) { plyRow(first, i); i += 1; continue; }
      if (!second) break;
      const stripe = first.round % 2 === 0;
      const number = SESSION.spy ? `round ${first.round + 1}` : first.round + 1;
      rows.push({ plies: i + 2, kind: 'move', stripe: stripe,
                  html: roundRowHtml(number, first, second) });
      i += 2;
      if (!(second.phase || []).length) continue;
      const beat = split.get(i);
      rows.push({ plies: i, kind: 'phase', stripe: stripe,
                  took: second.phase.some(step => step.eaten),
                  html: esc(phaseText(second.phase, beat ? beat.walkers : null)) });
    }
    if (outcome) rows.push({ plies: last + 1, kind: 'result', stripe: false,
                             html: `<b>${esc(verdictText(outcome))}</b>` });
    return rows;
  }
  // Generation 42 (N4c, §4.6): a duck game writes a turn on one line, its piece move and
  // the duck's half (`duckRowHtml`); the review still steps a ply at a time, and either ply
  // of the turn finds its row. The wild pieces' walk, after the duck's half, goes under it.
  if (SESSION.ruleset.duck !== null && SESSION.ruleset.duck !== undefined) {
    let i = 0, turn = 0;
    while (i < SESSION.history.length) {
      const m = SESSION.history[i];
      const duck = SESSION.history[i + 1] && SESSION.history[i + 1].pass !== undefined
        ? SESSION.history[i + 1] : null;
      const end = duck ? i + 1 : i;
      const stripe = turn % 2 === 0;
      rows.push({ plies: end + 1, also: i + 1, kind: 'move', stripe: stripe, html: `${
        stripe ? `${Math.floor(turn / 2) + 1}.` : '&nbsp;&nbsp;&hellip;'} ${duckRowHtml(m, duck)}${
        signHtml(duck || m, end === last ? outcome : null)}${liftedHtml(m)}${lookedHtml(m)}${
        corrMovedDayHtml(i)}` });
      const walked = (duck || m).phase || [];
      if (walked.length) {
        const beat = split.get(end + 1);
        rows.push({ plies: end + 1, kind: 'phase', stripe: stripe,
                    took: walked.some(step => step.eaten),
                    html: esc(phaseText(walked, beat ? beat.walkers : null)) });
      }
      i = end + 1;
      turn += 1;
    }
    if (outcome) rows.push({ plies: last + 1, kind: 'result', stripe: false,
                             html: `<b>${esc(verdictText(outcome))}</b>` });
    return rows;
  }
  SESSION.history.forEach(plyRow);
  // The game's end, in the strip's own words, as the list's last line.
  if (outcome) rows.push({ plies: last + 1, kind: 'result', stripe: false,
                           html: `<b>${esc(verdictText(outcome))}</b>` });
  return rows;
}

// The sign after a move that left a royal piece attacked: the session's `check` for that
// move, which is the core's answer (`web/play.py`). `#` where that move ended
// the game by checkmate, as the engine reported the outcome; `+` otherwise. A
// game with no royal piece has no check, and its moves carry no sign.
function signHtml(move, outcome) {
  if (!move || !move.check) return '';
  const mate = !!(outcome && outcome.reason === 'checkmate');
  return `<span class="sign" title="${mate ? 'checkmate: the game ended here'
    : 'check: this move attacked a royal piece'}">${mate ? '#' : '+'}</span>`;
}

// A phase in words, with the spec's names for the pieces (`phaseWords`, shared with the
// offline replay).
function phaseText(steps, walkers) {
  const rules = SESSION.ruleset;
  const nameOf = value => {
    const piece = pieceOf(value);
    return piece ? (rules.pieceNames || [])[piece.type] || (rules.pieceIds || [])[piece.type]
                 : null;
  };
  // Generation 40 (N3c): the compiled pounce kind of a wild piece's type, for its words.
  const pounceOf = value => {
    const piece = pieceOf(value);
    return piece ? (rules.pounceKinds || [])[piece.type] || null : null;
  };
  return phaseWords(steps, walkers, nameOf, rules.width, pounceOf);
}

// What to call a side, and who is sitting on it. A seat is an agent string --
// "minimax:nodes=1600" -- which is the identity the ladder rates and the record stores,
// and not what a person calls the thing beating them. The roster's `name` is what a
// sentence says (`shortName`); what the budget does belongs in the picker and under the
// seat, not here.
function sideName(player) {
  return (SESSION.ruleset.sides || [])[player] || `player ${player}`;
}

function seatName(agent) {
  return shortName(agent);
}

// Who sits in a seat, in words: the member's handle in a game between two people
// (Q570), and the seat's player name everywhere else.
function seatLabel(player) {
  const between = SESSION && SESSION.correspondence;
  if (between) {
    const handle = between.players[player];
    return between.you === player ? `${handle} (you)` : handle;
  }
  return seatName(SESSION.seats[player]);
}

// Under the name, how hard that seat thinks: the server's plain words off the budget it
// plays at (`web/play.py::strength`). A name alone left two machines reading "hard vs
// hard" with nothing to say what hard meant (Q288). Nothing under a person.
function seatHtml(player) {
  const strength = (SESSION.strength || [])[player] || '';
  // A person's seat carries their mark (R2) beside their handle.
  const between = SESSION.correspondence;
  const mark = between && between.marks ? markHtml(between.marks[player], 18) : '';
  // And links to their page (R1): only a member with a mark has one, never somebody who
  // left.
  const member = between && between.marks && between.marks[player];
  const label = member && between.players && between.players[player]
    ? `<a href="/players/${esc(between.players[player])}">${mark}${esc(seatLabel(player))}</a>`
    : `${mark}${esc(seatLabel(player))}`;
  return `<b>${esc(sideName(player))}</b>
    <span class="muted">${label}</span>${initiativeHtml(player)}${
    strength ? `<div class="muted small">${esc(strength)}</div>` : ''}`;
}

// Schema 29 (Q983b, `docs/plan-rest.md` §5.6): a move chosen under a lift, beside the move,
// from its entry's `lifted` (the core's answer, `web/play.py::rest_entry`). A round says it in
// its sentences instead (`round_reveal`'s words), so only a turn's row carries the mark.
function liftedHtml(move) {
  if (!move || !move.lifted || move.round !== undefined) return '';
  return ` <span class="muted small" data-lifted title="${esc(
    "every move this side had was a resting piece's, so the rest was lifted for this turn")}">${
    esc('rest lifted')}</span>`;
}

// How far the machine looked before it played a move, beside the move, as the core
// reported it with the move (`web/play.py::_reply`). Counted in turns, each player's move
// being one (§3.2's word for a ply), which is the unit the picker's greedy player is
// described in too; the positions a searching player looks at are what bought the turns,
// and the title says so, so the picker's number and this one are read as one thing (F7,
// P14). Nothing for a person's move, a player that does not search ahead, or a move that
// was the only one.
function lookedHtml(move) {
  const depth = move && move.depth;
  if (depth === null || depth === undefined) return '';
  // Schema 26: a round player looks ahead in rounds, both players' moves at once (Q918b).
  if (move.round !== undefined) {
    const rounds = `${depth} round${depth === 1 ? '' : 's'}`;
    return ` <span class="muted small" title="${esc(`it weighed every pair of moves ${rounds} ahead before it chose this one`)}">${
      esc(`looked ${rounds} ahead`)}</span>`;
  }
  const turns = `${depth} turn${depth === 1 ? '' : 's'}`;
  const said = depth === 0 ? 'played without looking ahead' : `looked ${turns} ahead`;
  const why = depth === 0
    ? 'it ran out of positions to look at before it had looked even one turn ahead, so it played a move it had not compared with the others'
    : `the positions it looks at before each move took it ${turns} ahead here; a turn is one player's move, its own or the reply`;
  return ` <span class="muted small" title="${esc(why)}">${esc(said)}</span>`;
}

// A move's depth arrives once, with the reply that played it. The site rebuilds a game
// from its stored moves on every request, and since Q424 it stores each depth beside its
// move (`matches.depths`, `0014_depths.sql`), so a reload says what the board said. What
// this is still for is a page left open across that deploy: a row from before it kept no
// depths, its next reply writes null for every move before that one, and the rebuilt
// game comes back without the depths this page had already been told -- so the page keeps
// them for the moves it is still showing. Nothing is worked out: a depth the page was
// never told stays unsaid.
function keepDepths(before, after) {
  if (!before || !after || before.id !== after.id) return;
  (after.history || []).forEach((move, i) => {
    const had = (before.history || [])[i];
    if (move.depth === undefined || move.depth === null) {
      if (had && had.depth !== undefined && had.depth !== null
          && had.ply === move.ply && had.text === move.text && had.by === move.by) {
        move.depth = had.depth;
      }
    }
  });
}

// What kind of game this is, under the two seats. A rated game and a sandbox game used
// to look identical, so a person could not tell which of them they were in until a
// take-back was refused -- and on the site that is the difference between a result that
// moves the ladder and one that counts for nothing.
//
// Only the store has modes. The local application sends none and gets no mark, because
// a mark that is always the same says nothing.
function modeHtml(snapshot) {
  if (snapshot.correspondence) {
    const rated = snapshot.correspondence.rated;
    return `<div class="u-mt-6"><span class="tag" title="${rated
      ? 'a game between two people; both ratings on this game move at the end'
      : 'a game between two people on rules with no rated computer players; it is kept and counts for nothing'
    }">${rated ? 'rated' : 'practice'}</span></div>`;
  }
  if (snapshot.mode === 'leaderboard') {
    return `<div class="u-mt-6"><span class="tag u-bc-good u-c-good" title="this result is rated, there is no take-back, and leaving
      resigns the game">rated</span></div>`;
  }
  if (snapshot.mode === 'sandbox') {
    return `<div class="u-mt-6"><span class="tag"
      title="undo and restart are on and nothing here counts towards a rating"
      >sandbox</span></div>`;
  }
  return '';
}

// The engine's reasons for a game ending (`Reason::as_str` in the core), as the end of a
// sentence that starts *‹side› wins* or *Drawn*. Every reason the core has is here, and
// one it gains later reads *by the rules* until it is: a reason's key is never shown
// (P19). `resigned` is the store's word, not the engine's: a thing somebody did rather
// than a way a position ended.
const REASON_WORDS = {
  checkmate: 'by checkmate', stalemate: 'by stalemate', stalemate_loss: 'by stalemate',
  stalemate_win: 'by stalemate', royal_captured: 'by capturing the royal piece',
  royal_destroyed: 'by destroying the royal piece', annihilated: 'by taking every piece',
  repetition: 'by repetition', no_progress: 'by the no-progress rule',
  ply_limit: 'at the move limit', reached: 'by reaching the goal',
  extinction: 'by wiping out a whole kind of piece',
  bare_royal: 'by leaving the royal piece alone',
  check_limit: 'by giving check enough times', coloured: 'on ground held',
  scored: 'on points', resigned: 'by resignation', agreed: 'by agreement',
  // The store's too (Q918f): a game in rounds where neither player sealed a move in time.
  unsealed: 'as neither player sealed a move in time',
};

function reasonWords(reason) {
  // Schema 18 (Q786): a game won by moving off its goal square ends on the reach rule's
  // reason, so the words follow the compiled rule the session carries (`goalLeaves`).
  if (reason === 'reached' && SESSION && SESSION.ruleset && SESSION.ruleset.goalLeaves) {
    return 'by leaving the goal';
  }
  return REASON_WORDS[reason] || 'by the rules';
}

// How the game ended, in one sentence: the engine's outcome, or the store's for a game
// somebody gave up. The status strip says it, directly above the board, and the move
// list ends on it.
function verdictText(outcome) {
  const parts = verdictParts(outcome);
  return parts.how ? `${parts.head} ${parts.how}.` : parts.head;
}

// The same sentence in its two halves, for the ended banner over the board, which sets
// the result in the heading face and the engine's reason under it: *who won* and *how*.
function verdictParts(outcome) {
  if (outcome.reason === 'cancelled') {
    return { head: 'Cancelled before both players had moved. It counts for nothing.', how: '' };
  }
  // Schema 27 (Q942b): an ending that fired for both players in one round carries the
  // server's own words for it (`play.both_fired_words`), since *by capturing the royal
  // piece* would name one capture where there were two.
  const how = outcome.how || reasonWords(outcome.reason);
  if (outcome.winner === null || outcome.winner === undefined) return { head: 'Drawn', how: how };
  const side = sideName(outcome.winner);
  return { head: `${side.charAt(0).toUpperCase()}${side.slice(1)} (${
    seatLabel(outcome.winner)}) wins`, how: how };
}

// Whose result it is, for the banner's colour: the reader's own side's win or loss, and
// neither where the reader sat on no side or on both (a watched game, two people at one
// screen, a draw). The seats the store and the session name, never worked out.
function resultFor(outcome) {
  if (outcome.winner === null || outcome.winner === undefined) return '';
  const between = SESSION.correspondence;
  const you = between ? between.you
    : SESSION.seats.filter(seat => seat === 'human').length === 1
      ? SESSION.seats.indexOf('human') : null;
  if (you === null || you === undefined) return '';
  return outcome.winner === you ? 'won' : 'lost';
}

// The ended state (plan-visual.md §3.3): a banner over the middle of the board, the board
// still to be seen round it, with the result, the engine's reason, and the two things a
// finished game offers -- the game again, by this board's own control for it and in its
// words, and a look back through it, which steps into the review the row under the board
// already offers. Only on the game as it stands: stepping back takes it away.
function drawEnded() {
  const banner = document.getElementById('ended');
  if (!banner || !SESSION) return;
  const outcome = SESSION.outcome;
  if (!outcome || AT) { banner.hidden = true; banner.innerHTML = ''; return; }
  const parts = verdictParts(outcome);
  const again = ['again', 'rematch'].map(id => document.getElementById(id))
    .find(node => node && !node.hidden);
  // Q993b: on the proven positions' own page the banner's primary is *next position*, and
  // the try again sits beside it (S6: one primary, naming its result).
  const next = BOARD_PROVEN ? document.getElementById('provenNext') : null;
  const key = `${outcome.winner}|${outcome.reason}|${again ? again.id : ''}|${
    next && !next.disabled ? 'next' : ''}|${TIMELINE.length}`;
  banner.hidden = false;
  if (banner.dataset.built === key) return;
  banner.dataset.built = key;
  const word = again ? (again.querySelector && again.querySelector('.w')
    ? again.querySelector('.w').textContent : again.textContent).trim() : '';
  banner.className = `frame-ended ${resultFor(outcome)}`;
  const going = next && !next.disabled ? next.textContent.trim() : '';
  banner.innerHTML = `<div class="ended-head">${esc(parts.head)}</div>${
    parts.how ? `<div class="ended-how">${esc(parts.how)}</div>` : ''}<div class="row">${
    going ? `<button class="primary" data-ended="next">${esc(going)}</button>` : ''}${
    again ? `<button class="${going ? '' : 'primary'}" data-ended="again">${esc(word)}</button>` : ''}${
    TIMELINE.length > 1 ? '<button class="quiet" data-ended="review">review</button>' : ''}</div>`;
  banner.querySelectorAll('[data-ended]').forEach(node => {
    node.onclick = () => {
      if (node.dataset.ended === 'review') stepBy(-1);
      else if (node.dataset.ended === 'next') next.click();
      else if (again) again.click();
    };
  });
}

// Whether the position on screen has a royal piece attacked, as the engine said it: the
// snapshot's `inCheck` for the game as it stands, and for a position the game has left,
// the `check` the session recorded with the move that made it (`web/play.py`, asked of
// the core as the move was played). A ply with a wild-piece phase is in check only once
// the phase has run, which is the position the core was asked about. A finished game is
// reported with `inCheck` false, so its last move says whether it left the loser's royal
// piece attacked -- which is what a checkmate looks like.
function checkShown() {
  if (!SESSION) return false;
  if (AT) {
    const move = AT.lastMove;
    return !!(move && move.check && !(AT.kind === 'move' && (move.phase || []).length));
  }
  if (SESSION.inCheck) return true;
  return !!(SESSION.outcome && SESSION.lastMove && SESSION.lastMove.check);
}

// Where to draw that check: the square of the side to move's royal piece, found by the
// compiled `royal` flags on the position on screen. Nothing is decided here -- the engine
// said the side is in check, and this only finds where its royal piece stands. A side
// holding several royal pieces gets no square, because which of them is attacked is the
// engine's to say and it says only that one is; the status strip says *in check* anyway.
function checkSquares() {
  if (!checkShown()) return [];
  const royal = SESSION.ruleset.royal || [];
  const side = AT ? AT.sideToMove : SESSION.sideToMove;
  const found = [];
  position().forEach((value, square) => {
    const piece = pieceOf(value);
    if (piece && piece.player === side && royal[piece.type]) found.push(square);
  });
  return found.length === 1 ? found : [];
}

// The tab a board opens on, where something other than the moves asked (K3).
let OPEN_TAB = null;

// K3 (`docs/archive/plan-understanding.md` §4.2 step 4): a tapped piece's move sentences in one
// line under the board, the *pieces* tab's own words (`pieces[type].moves`, already sent
// with the game), so a newcomer need not switch tabs to ask what this one does.
function drawPieceSays() {
  const line = document.getElementById('pieceSays');
  if (!line || !SESSION) return;
  // Schema 19: a portal end picked or tapped says what it does, in the words its label
  // and its square's tooltip carry (`portalWords`), and how a piece comes out of it.
  const looked = SELECTED !== null && !AT && portalOn(SELECTED) ? SELECTED : INSPECT;
  const portal = looked !== null ? portalOn(looked) || foldOn(looked) : null;
  if (portal) {
    line.hidden = false;
    line.innerHTML = `<b>${esc(portal.words.head)}</b>. ${esc(portal.words.says)}`;
    sayMore(line);
    return;
  }
  const game = DETAIL[SESSION.ruleset.hash];
  // The piece picked, else (schema 30) a piece tapped to see what it sees, either side's.
  const tapped = sightTapped();
  const chosen = SELECTED !== null && !AT ? SELECTED : tapped;
  const piece = chosen !== null ? pieceAt(chosen) : null;
  const about = piece && game && (game.pieces || [])[piece.type];
  // Its ways of moving, then what its lines of sight do (`describe`'s sentences, Q999c), then
  // (schema 29, Q983b) its rest and, resting now, the turns it has left.
  const said = about
    ? (about.moves || []).concat((about.sights || []).map(sight => sight.words),
                                 pieceRestLine(about, chosen)) : [];
  line.hidden = !said.length;
  line.innerHTML = said.length
    ? `<b>${esc(about.name || about.id)}</b>: ${said.map(esc).join('; ')}` : '';
  sayMore(line);
}

// Schema 29 (Q983b, `docs/plan-rest.md` §5.6): a piece's rest in words for the line under a
// picked piece, from its card's `rest` (the compiled table's) and the position's `resting`
// (the core's): *rests 2 turns after it moves*, and *resting: 1 more turn* while it rests.
function pieceRestLine(about, square) {
  const turns = about && about.rest;
  if (!turns) return [];
  const said = [`rests ${turns} turn${turns === 1 ? '' : 's'} after it moves`];
  const resting = positionRests(AT ? AT.resting : SESSION.resting)
    .find(rest => rest.square === square);
  if (resting && !SESSION.outcome) {
    said.push(`resting: ${resting.left} more turn${resting.left === 1 ? '' : 's'}`);
  }
  return said;
}

// Q809: the sentences are held to the one line `fitBoard` keeps for them (the style's
// `.piece-says`), so a piece whose sentences run longer pushes nothing off the screen when
// it is picked. Where they are cut, the line is a button that shows them whole, and again
// one line; the reader's choice holds for those words until other words are shown. The
// board is the same size either way: `fitBoard` never reads the line's height.
let SAYS_OPEN = null;
function sayMore(line) {
  if (!line.classList || typeof line.setAttribute !== 'function'
      || typeof line.removeAttribute !== 'function') return;
  const words = line.hidden ? '' : String(line.textContent || '');
  if (SAYS_OPEN !== words) SAYS_OPEN = null;
  const opened = !!words && SAYS_OPEN === words;
  line.classList.toggle('open', opened);
  const cut = !!words && !opened && typeof line.scrollHeight === 'number'
    && line.scrollHeight > line.clientHeight + 1;
  const more = cut || opened;
  line.classList.toggle('more', more);
  if (more) {
    line.setAttribute('role', 'button');
    line.setAttribute('tabindex', '0');
    line.setAttribute('aria-expanded', String(opened));
    line.title = opened ? 'show one line of it' : 'show all of it';
  } else {
    for (const name of ['role', 'tabindex', 'aria-expanded', 'title']) line.removeAttribute(name);
  }
  const flip = () => { SAYS_OPEN = opened ? null : words; sayMore(line); };
  line.onclick = more ? flip : null;
  line.onkeydown = more ? event => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); flip(); }
  } : null;
}

// The words *in check*, as a pill of their own in the status strip, in the alarm colour
// the board's check mark is drawn in.
function checkPill() {
  return ' <span class="pill alarm" title="a royal piece is under attack">in check</span>';
}

// A line the status strip carries until the next click or move: what a click that did
// nothing meant. Today one sentence, for a piece with no legal move.
let NOTE = '';

function thinkingHtml() {
  const mover = SESSION ? SESSION.sideToMove : 0;
  const slow = SESSION && AGENT_SLOW === SESSION.id
    ? `<span class="frame-note">${esc(SLOW_SAID)}</span>` : '';
  // In a round (schema 26) the computer is choosing its sealed move, unseen; in a spy
  // game's announced round (schema 27), its announcement or its answer to one.
  const round = SESSION && SESSION.round;
  let doing = round ? ' is choosing its move' : ' is thinking';
  if (round && round.announcer !== undefined && round.announcer !== null) {
    doing = round.announcer === mover ? ' is choosing what to announce'
      : round.announced ? ` is choosing its answer to ${round.announced.text}`
      : ' is choosing its answer';
  }
  return `<span class="spin"></span> ${SESSION ? seatDot(mover) + esc(sideName(mover)) + ' (' +
    esc(seatName(SESSION.seats[mover])) + ')' + esc(doing) : 'thinking'}${slow}${
    SESSION ? spyNoteHtml() : ''}`;
}

// The side to move as a dot in its army's colour, before its name: the strip's first
// word, read before the words are.
function seatDot(player) {
  return `<span class="seat-dot s${player === 1 ? 1 : 0}" aria-hidden="true"></span>`;
}

function drawStatus() {
  drawStrip();
  drawEnded();
}

function drawStrip() {
  const status = document.getElementById('status');
  const ground = document.getElementById('ground');
  if (!status) return;
  // The strip is the frame's first slot; the result turns the whole of it into the
  // verdict, directly above the board, where every reader was already looking.
  const strip = status.closest ? status.closest('[data-frame="status"]') : null;
  const ended = on => { if (strip && strip.classList) strip.classList.toggle('ended', on); };

  if (ground) ground.innerHTML = groundHtml();
  // The slot under the board that sends a move between two people says the game as it
  // stands, whichever position the board is showing.
  drawCorrSend();
  const undo = document.getElementById('undo');
  // A game that is over cannot be resigned, and neither can a position the game has
  // already left -- the store refuses both, and the button says so before it is pressed.
  const resign = document.getElementById('resign');
  if (resign) resign.disabled = !!AT || !!SESSION.outcome;
  // A draw offered to the computer (Q790) is made on the person's own turn, of the game
  // as it stands: not while the computer thinks, not over a reviewed position.
  const offerDraw = document.getElementById('draw');
  if (offerDraw && !SESSION.correspondence) {
    offerDraw.disabled = !!AT || !!SESSION.outcome || !SESSION.drawOffer
      || !!SESSION.awaitingAgent;
  }

  // A reviewed position is one the game has already left, so neither whose move it is
  // nor how the game ended is true of what is on screen. Saying either here would be
  // describing a different board from the one being looked at; the review row says
  // where you are instead, and taking a move back is not something you can do to a
  // position you have stepped away from.
  // The strip says it is an earlier position, and whether that position had a royal
  // piece attacked; the review row under the board says which one it is.
  if (AT) {
    ended(false);
    status.innerHTML = 'an earlier position' + (checkShown() ? checkPill() : '');
    if (undo) undo.disabled = true;
    return;
  }

  if (SESSION.outcome) {
    ended(true);
    status.textContent = verdictText(SESSION.outcome);
  } else if (SESSION.correspondence) {
    ended(false);
    status.innerHTML = corrStatus();
  } else if (SESSION.round && !SESSION.awaitingAgent) {
    ended(false);
    status.innerHTML = roundStatusHtml();
  } else {
    ended(false);
    // Generation 42 (N4c, §4.6): a duck game's turn is two halves, said as what to do next,
    // never as moves left.
    const more = SESSION.ruleset.duck !== null && SESSION.ruleset.duck !== undefined
      ? (SESSION.duckOwed ? ' - now move the duck' : '')
      : SESSION.remaining > 1 ? ` - ${SESSION.remaining} moves left this turn` : '';
    const check = SESSION.inCheck ? checkPill() : '';
    // "thinking" is also written by `maybeAgent` as it asks. A redraw while the answer
    // is on its way (a flip, a toggle) says it again rather than blanking the strip, and
    // a paused game with a machine to move says so, because `maybeAgent` is not going
    // to be called while the game is stopped.
    status.innerHTML = SESSION.awaitingAgent
      ? (PAUSED ? 'paused' : AGENT_ASKING === SESSION.id ? thinkingHtml() : '') + check
      : `${seatDot(SESSION.sideToMove)}${esc(sideName(SESSION.sideToMove))} (${
          esc(seatLabel(SESSION.sideToMove))}) to move${check}${esc(more)}${spyNoteHtml()}${
          NOTE ? `<span class="frame-note">${esc(NOTE)}</span>` : ''}`;
  }
  if (undo) undo.disabled = SESSION.history.length === 0;
}

// The score, where the game has one on the board itself. A territory game is won by a
// count of squares, and a player who cannot see the count is playing blind.
function groundHtml() {
  const rule = SESSION.ruleset.territory;
  const colour = tint();
  if (!rule || !colour.length) return '';
  let first = 0, second = 0;
  for (const c of colour) { if (c === 1) first++; else if (c === 2) second++; }
  const target = rule.win_at ? ` - ${rule.win_at} claims the game` : '';
  // Named, because the two numbers used to be "12 to 9" with nothing saying which was
  // whose -- and on a flipped board the reader cannot even guess from the picture.
  return `ground held: ${esc(sideName(0))} <b>${first}</b>, ${esc(sideName(1))} <b>${second}</b>
    of ${colour.length}${esc(target)}
    <span class="muted"> (${rule.mode === 'repaint'
      ? 'a claim can be taken back' : 'a claim is permanent'})</span>`;
}

function mark(x, y, cell, colour, opacity) {
  return cellShape(x, y, cell, FRAME,
    `fill="${colour}" opacity="${opacity}" pointer-events="none"`);
}

// Where a wild piece walked this ply: dashed on the square it left, solid on the one it
// stepped onto, both in the third owner's colour. Two weights rather than two colours,
// so the pair reads as a direction on a board already using amber for the last move.
function trail(x, y, cell, arrived) {
  const dash = arrived ? '' : ` stroke-dasharray="${cell * 0.13} ${cell * 0.09}"`;
  return cellShape(x, y, cell, FRAME, `fill="none" stroke="var(--p2)" stroke-width="${cell * 0.07}"
           opacity="${arrived ? 0.95 : 0.7}"${dash} pointer-events="none"`, 0.06);
}

function movesFrom(square) {
  if (square === null || !SESSION) return [];
  return SESSION.moves.filter(m => m.from === square);
}

function drawHand() {
  const hand = document.getElementById('hand');
  if (!hand) return;
  // A hand is a count per piece type, the way the state carries it -- index 2 holding 3
  // means three of piece type 2, not one piece called 3. While reviewing it is the hand
  // *that* position had: the live one belongs to a board nobody is looking at, and a
  // captured piece appearing in it twenty plies early is the same kind of lie as a piece
  // standing on the wrong square.
  const owner = AT ? AT.sideToMove : SESSION.sideToMove;
  const mine = ((AT ? AT.hands : SESSION.hands) || [])[owner] || [];
  const held = mine.map((count, type) => [type, count]).filter(([, count]) => count > 0);
  const droppable = new Set(SESSION.moves.filter(m => m.drop !== undefined).map(m => m.drop));
  // Generation 42 (N4c): with the duck owed, the strip is the duck's: the duck waiting off the
  // board, or the pass where it is the one move.
  const tray = duckTrayHtml();
  if (tray) { hand.innerHTML = tray; wireDuckTray(hand); return; }
  if (!held.length) { hand.innerHTML = ''; return; }

  // Schema 18e (Q785): a held piece may come back as its other face, a second button on
  // the same count. `dropFaces` is the compiled pair per slot (-1 for none), and `DROP`
  // is the face chosen, which is what each drop move names.
  const faces = SESSION.ruleset.dropFaces || null;
  const ids = SESSION.ruleset.pieceIds;
  const buttons = [];
  held.forEach(([type, count]) => {
    const shown = faces && faces[type] >= 0 ? [type, faces[type]] : [type];
    shown.forEach(face => {
      const other = face !== type;
      buttons.push(`<button data-drop="${face}" class="${DROP === face ? 'on' : ''}"
        ${other ? `title="${esc(ids[type])} in hand, played as its other face"` : ''}
        ${!AT && droppable.has(face) ? '' : 'disabled'}>${esc(ids[face])}${other ? ` (${esc(ids[type])})` : ''} x${count}</button>`);
    });
  });
  hand.innerHTML = '<span class="muted u-as-center">in hand:</span>' +
    buttons.join('');
  hand.querySelectorAll('[data-drop]').forEach(node => {
    node.onclick = () => {
      DROP = DROP === Number(node.dataset.drop) ? null : Number(node.dataset.drop);
      SELECTED = null;
      drawBoard();
    };
  });
}

// ----- the duck (generation 42, N4c, `docs/plan-npc.md` §4.6)
//
// A duck game's turn is two plies: the piece move, picked as any move is, and then the same
// player's duck move. Between them the board is in duck mode: every square the duck may go
// to is lit at once (there is one duck, so the lit squares are the whole choice), its own
// square is ringed, and the strip says *now move the duck*. Off the board, the duck waits in
// the hand's strip. The pass is a button, shown only when it is the one legal move: the page
// never plays a person's move for them. Every target is the core's (`SESSION.moves`).
function duckMode() {
  if (!SESSION || AT || !SESSION.duckOwed || SESSION.outcome || SESSION.awaitingAgent) return false;
  if (SESSION.seats[SESSION.sideToMove] !== 'human') return false;
  const between = SESSION.correspondence;
  return !(between && between.you !== SESSION.sideToMove);
}

function duckTargets() {
  if (!duckMode()) return new Set();
  return new Set(SESSION.moves.filter(m => m.kind === 'duck' && !m.pass).map(m => m.to));
}

function duckPassMove() {
  return duckMode() ? SESSION.moves.find(m => m.kind === 'duck' && m.pass) || null : null;
}

// The duck's own drawing, as the board draws it (a piece nobody owns).
function duckPieceSvg() {
  const rules = SESSION.ruleset;
  const stride = rules.sideStride || rules.nTypes;
  const piece = pieceOf(2 * stride + rules.duck + 1);
  return `<svg class="duck-tray" viewBox="0 0 10 10" aria-hidden="true">${
    drawPiece(piece, 0, 0, 10)}</svg>`;
}

function duckTrayHtml() {
  if (!duckMode()) return '';
  const pass = duckPassMove();
  if (pass) {
    return `<button class="duck-pass" data-duck-pass="${pass.move}">${
      'the duck has nowhere to go: pass'}</button>`;
  }
  if (SESSION.duck !== null && SESSION.duck !== undefined) return '';
  return `${duckPieceSvg()}<span class="muted u-as-center">the duck waits off the board:
    click a lit square to put it there</span>`;
}

function wireDuckTray(hand) {
  hand.querySelectorAll('[data-duck-pass]').forEach(node => {
    node.onclick = () => send(Number(node.dataset.duckPass));
  });
}

// The duck's flight on a ply (`last`, a history entry): a short solid arrow of its own olive
// from its old square to its new one, over the board; nothing for a placement from off the
// board, a pass or any other move. Off the entry's own squares, never a board diff.
function duckFlightHtml(last, cell) {
  if (!last || last.pass === undefined || last.from === null || last.to === null) return '';
  const a = squareXY(last.from, cell, FRAME), b = squareXY(last.to, cell, FRAME);
  const ax = a[0] + cell / 2, ay = a[1] + cell / 2, bx = b[0] + cell / 2, by = b[1] + cell / 2;
  const length = Math.hypot(bx - ax, by - ay);
  if (!(length > cell * 0.6)) return '';
  const ux = (bx - ax) / length, uy = (by - ay) / length;
  return announceLine(ax + ux * cell * 0.3, ay + uy * cell * 0.3, bx - ux * cell * 0.3,
                      by - uy * cell * 0.3, cell, 'duckFlight', 'var(--duck)');
}

// A duck game's turn on one line of the move list (§4.6): *e4 · duck d5*, the duck's half in
// its square's name, or *passes* where it had nowhere to go.
function duckRowHtml(m, duck) {
  if (!duck) return esc(m.text);
  const W = SESSION.ruleset.width;
  const where = duck.pass ? 'passes'
    : squareName(duck.to % W, Math.floor(duck.to / W));
  return `${esc(m.text)} <span class="muted">&middot;</span> duck ${esc(where)}`;
}

function clickSquare(square) {
  // Schema 19: a tap on a portal end says what it does, under the board (D9), whoever's
  // turn it is and whatever position is on screen; a tap anywhere else puts that away.
  // A fold (schema 22) says what it does when nothing stands in it; a piece standing in one
  // is picked, as anywhere.
  const inspected = portalOn(square) || (foldOn(square) && !pieceAt(square)) ? square : null;
  if (inspected !== INSPECT && !PROMO) {
    INSPECT = inspected;
    drawPieceSays();
    paintLinks();
    fitBoard();
  }
  // Schema 30 (Q999c): a tap on a piece with a line of sight shows what it sees, and says
  // it under the board, whoever's piece it is; a tap anywhere else puts it away. A touch
  // screen has no pointer to rest on a piece, so this is the hover there.
  const tapped = !AT && !PROMO && sightOf(square) ? { square, ply: SESSION.ply } : null;
  if (JSON.stringify(tapped) !== JSON.stringify(SIGHT_TAP)) {
    SIGHT_TAP = tapped;
    paintSight();
    drawPieceSays();
  }
  // The board is showing a position the game has left; there is nothing to move in it.
  if (AT || PROMO) return;
  if (!SESSION || SESSION.outcome || SESSION.awaitingAgent) return;
  if (SESSION.seats[SESSION.sideToMove] !== 'human') return;
  // Between two people, only the player to move picks, and a picked move waits to be
  // sent -- and until it is sent it can be changed on the board (Q740): see `corrRepick`.
  const between = SESSION.correspondence;
  if (between && (between.you !== SESSION.sideToMove || CORR_SENDING)) return;
  // Schema 26: a move picked in a round is held the same way until it is sealed.
  if (SESSION.round && ROUND_SENDING) return;
  if ((between || SESSION.round) && CORR_HELD !== null && corrRepick(square)) return;
  NOTE = '';

  // Generation 42 (N4c, `docs/plan-npc.md` §4.6): with the duck owed, a click on a lit square
  // is that duck move, and every other click says what is owed. The moves are the core's.
  if (SESSION.duckOwed) {
    const duck = SESSION.moves.find(m => m.kind === 'duck' && !m.pass && m.to === square);
    if (duck) return send(duck.move);
    NOTE = duckPassMove() ? 'the duck has nowhere to go: pass it with the button under the board'
      : 'now move the duck: click one of the lit squares';
    drawBoard();
    return;
  }

  if (DROP !== null) {
    const drop = SESSION.moves.find(m => m.drop === DROP && m.to === square);
    if (drop) return send(drop.move);
    DROP = null;
  }

  // Schema 19: a tap on an end the picked piece can go in by is that way in (`takeWays`);
  // else a tap on a square it can reach is that move, or the choice among the moves that
  // reach it (`takeLanding`).
  const ways = movesFrom(SELECTED).filter(m => m.portal === square);
  const candidates = movesFrom(SELECTED).filter(m => m.to === square);
  // Schema 22: a piece standing in a fold its side may move, and a square both can reach --
  // the piece's move or the fold's own -- is a choice of which to move, in the user's words
  // (`docs/archive/plan-portals-2.md` §5.3R): *Move the piece* or *Move the portal*, never both.
  const folded = candidates.filter(m => m.kind === 'fold');
  if (folded.length && folded.length < candidates.length) {
    return choosePieceOrPortal(candidates.filter(m => m.kind !== 'fold'), folded, SELECTED, square);
  }
  // Q867c, the user's misplay (§5.3R, 2026-10-12): with a piece picked, a tap on a portal the
  // player to move may move, where that piece may also land or go in, meant *pick the
  // portal* as often as *land there*. It asks the same question, *Move the piece* or *Move
  // the portal*; every other landing stays one tap.
  const movable = SELECTED !== null && SELECTED !== square ? portalMovesFrom(square) : [];
  if (movable.length && (ways.length || candidates.length)) {
    return choosePieceOrPortal(ways.length ? ways : candidates, null, square, square);
  }
  if (takeWays(ways) || takeLanding(candidates)) return;

  // On the first click, if the piece has rotation moves, show the rotation chooser
  // immediately instead of requiring a second click on the same square.
  const fromHere = movesFrom(square);
  // Another of your pieces that can move is picked *instead* of the move held (Q740): the
  // held one goes, and this piece's squares show.
  if (fromHere.length && CORR_HELD !== null) corrForget();
  const rotations = fromHere.filter(m => m.rotation !== undefined);
  if (rotations.length > 0) {
    SELECTED = square;
    drawBoard();
    return chooseRotation(rotations);
  }

  SELECTED = fromHere.length ? square : null;
  // A click on one of your own pieces that has nowhere to go used to do nothing and say
  // nothing, and a reader in check took the board for frozen (study B1). The moves are
  // the engine's list, so *no legal move* is its answer; *in check* is its `inCheck`.
  const piece = pieceAt(square);
  NOTE = !fromHere.length && piece && piece.player === SESSION.sideToMove
    ? `that ${pieceWord(piece)} has no legal move${
        SESSION.inCheck ? ' - your side is in check' : ''}${portalsOnlyWords()}` : '';
  // Schema 20: a portal the player to move may move, picked just after it was moved, says
  // why the square it came from has no dot (Q838c). The bar is the core's (`portalBar`).
  NOTE = NOTE || barNote(square);
  drawBoard();
}

// A tap's ways in by one portal end (schema 19): the move itself where it comes out on one
// square, else the choice of where it comes out (D6). False when there are none.
function takeWays(ways) {
  if (!ways.length) return false;
  if (!ways.every(m => m.to === ways[0].to)) chooseExit(ways);
  else if (ways.length === 1) send(ways[0].move);
  else choosePromotion(ways);
  return true;
}

// A tap's moves onto one square: the move itself, else the choice among them. False where
// nothing is played or asked, and the tap picks instead.
function takeLanding(candidates) {
  if (candidates.length === 1) { send(candidates[0].move); return true; }
  if (candidates.length < 2) return false;
  // Schema 19: one square reached more than one way -- directly and through a portal, or
  // through two -- is a choice of route (`docs/archive/plan-portals.md` §2.3), each the engine's
  // own move named by its text.
  if (candidates.some(m => m.portal !== undefined)
      && new Set(candidates.map(m => m.portal === undefined ? -1 : m.portal)).size > 1) {
    chooseRoute(candidates);
  } else if (candidates[0].landing !== undefined) {
    // A strike that places its survivor: one move per landing square (and per promotion).
    chooseLanding(candidates);
  } else if (candidates[0].promotion !== undefined) {
    // Several moves share these two squares: a promotion, and the player picks what to
    // become. Anything else sharing a square pair would be a bug in the generator.
    choosePromotion(candidates);
  } else if (candidates[0].rotation !== undefined) {
    chooseRotation(candidates);
  } else {
    return false;
  }
  return true;
}

// The moves of the portal standing on a square that the player to move may make now: a
// momentum end's (it is the piece there), or a fold's own (`MOVE_FOLD`) from under any piece
// standing in it. Read off the engine's list.
function portalMovesFrom(square) {
  if (portalOn(square)) return movesFrom(square);
  return foldOn(square) ? movesFrom(square).filter(m => m.kind === 'fold') : [];
}

function barNote(square) {
  const at = portalOn(square);
  if (!at || !(at.end.movedBy || []).includes(SESSION.sideToMove)) return '';
  const W = SESSION.ruleset.width;
  return portalBarWords(SESSION.portalBar, square, at.end, portalPairs(SESSION.ruleset.portals),
                        left => squareName(left % W, Math.floor(left / W)));
}

function pieceWord(piece) {
  const rules = SESSION.ruleset;
  return (rules.pieceNames || [])[piece.type] || (rules.pieceIds || [])[piece.type] || 'piece';
}

// Q833: when every move left is a portal end's, the note says so and names the ends. A
// player in check on a game whose portals both sides may move tapped each of their pieces,
// read *no legal move* on every one and took the game for stalled, when moving a portal
// onto the checking line was the way out. The ends are read off the engine's own list,
// as the rest of the note is; nothing here works a move out.
function portalsOnlyWords() {
  const from = [...new Set((SESSION.moves || []).map(m => m.from))];
  if (!from.length || !from.every(square => portalOn(square))) return '';
  const W = SESSION.ruleset.width;
  const names = from.sort((a, b) => a - b)
    .map(square => squareName(square % W, Math.floor(square / W)));
  const list = names.length === 1 ? names[0]
    : `${names.slice(0, -1).join(', ')} or ${names[names.length - 1]}`;
  return `; only a portal can move: tap the one on ${list}`;
}

// ----- what a promoting piece becomes (Q765)
//
// Two members each took the game for stalled when a pawn reached the far side: the choice
// of what it becomes was a row of small buttons under the board, out of sight on a phone.
// So the choice is drawn *over* the board, which is where the reader is looking when
// they make the move: the board dimmed behind a card that says, in plain words, what to
// do, with each piece drawn large on a square of the board's own wood and its name under
// it. The card is scrolled into view and takes the keyboard's focus; Tab stays inside it,
// the arrow keys walk the pieces, Enter or Space picks one, and Escape or *Cancel* puts
// the choice away with nothing moved: the pawn still on its square, nothing held or sent.
//
// **The position stays in reach while choosing** (the user, after the first cut): a
// person thinks about what the pawn should become by looking at the board, and a card
// over it hides the position. *Look at the board* puts the card and the dimming away and
// leaves one small bar under the board -- *Choose the new piece* to have the card back,
// and *Cancel* -- so every square can be seen while the move is still not made. The board
// takes no tap meanwhile: the move waits for its piece.
//
// A pick is the move, exactly as a click on a square is: `send`, so against the machine
// it is played, and between two people it is held until *Send your move* like any other
// pick (Q733), and can still be changed (Q740). The pieces offered are the engine's
// moves for that square pair, never a list the page makes up.
let PROMO = null;          // the moves on offer while the choice is open, else null
let PROMO_LOOKING = false; // the card put aside to look at the board, the choice still open
let PROMO_PIECE = null;    // a strike's: the piece type already picked, the square still to pick

function promotionWord(piece) {
  const names = SESSION.ruleset.pieceNames || [];
  return (piece && names[piece.type]) || 'piece';
}

function choosePromotion(candidates) {
  if (!document.getElementById('promo')) return;
  PROMO = candidates;
  PROMO_LOOKING = false;
  PROMO_PIECE = null;
  PROMO_ROUTE = false;
  PROMO_WHICH = null;
  drawPromotion();
}

// Schema 19: the way to a square reached through a portal and some other way. The same
// card, one button per move, named by the engine's text (`Bc1-d2>e3`, `Bc1-e3`); P2 draws
// the routes on the board.
let PROMO_ROUTE = false;
let PROMO_EXIT = false;
function chooseRoute(candidates) {
  if (!document.getElementById('promo')) return;
  PROMO = candidates;
  PROMO_LOOKING = false;
  PROMO_PIECE = null;
  PROMO_ROUTE = true;
  PROMO_WHICH = null;
  drawPromotion();
}

// Schema 22 and Q867c (`docs/archive/plan-portals-2.md` §5.3R): the piece or the portal, in the user's
// words, *Move the piece* or *Move the portal*, never both at once (no move carries a piece
// and its fold together; the core holds that). Asked where one tap means either: a piece
// standing in a fold its side may move, both reaching `to` (`portal` the fold's own moves
// there, played as they are); or a piece picked and a tap on a portal its side may move,
// where the piece may land or go in (`portal` null: *Move the portal* picks it, to choose
// where it goes). `at` is the portal's square, `piece` the piece's moves.
let PROMO_WHICH = null;
function choosePieceOrPortal(piece, portal, at, to) {
  if (!document.getElementById('promo')) return;
  PROMO = piece.concat(portal || []);
  PROMO_LOOKING = false;
  PROMO_PIECE = null;
  PROMO_ROUTE = false;
  PROMO_EXIT = false;
  PROMO_WHICH = { piece: piece, portal: portal, at: at, to: to };
  drawPromotion();
}

// The answer: the piece's move (or the choice among its moves), or the portal's own move,
// or the portal picked.
function pickWhich(which) {
  const chosen = PROMO_WHICH;
  closePromotion();
  if (!chosen) return;
  if (which === 'portal' && !chosen.portal) {
    SELECTED = chosen.at;
    NOTE = '';
    drawBoard();
    return;
  }
  const moves = which === 'portal' ? chosen.portal : chosen.piece;
  const ways = moves.filter(m => m.portal === chosen.to);
  if (!(ways.length ? takeWays(ways) : takeLanding(moves))) drawBoard();
}

// Schema 19: where a piece comes out of a portal it can reach two ways (D6, §2.3): one
// button a square, named by the way out the engine says it takes (`exitDir`) as the
// board is drawn, and the square. The bar under the board rather than the card over it,
// so the exit each button names can be lit on the board (`lightExit`, N6).
function chooseExit(candidates) {
  if (!document.getElementById('promo')) return;
  PROMO = candidates;
  PROMO_LOOKING = false;
  PROMO_PIECE = null;
  PROMO_ROUTE = false;
  PROMO_WHICH = null;
  PROMO_EXIT = true;
  drawPromotion();
}

function exitWords(m) {
  const W = SESSION.ruleset.width;
  const at = portalOn(m.portal);
  const name = at ? portalName(at.end, portalPairs(SESSION.ruleset.portals)) : 'the portal';
  const square = squareName(m.to % W, Math.floor(m.to / W));
  const step = portalStep(m.exitDir, FLIPPED);
  const words = step ? `${step.words} of ${name}, ${square}` : `beside ${name}, ${square}`;
  return { arrow: step ? step.arrow : '', words: words.charAt(0).toUpperCase() + words.slice(1) };
}

function drawExits(box) {
  const ways = [];
  for (const m of PROMO) if (!ways.some(way => way.to === m.to)) ways.push(m);
  const still = SESSION.correspondence ? ' You can still change your move until you send it.' : '';
  box.hidden = false;
  box.onkeydown = promotionKeys;
  if (box.classList) box.classList.toggle('looking', true);
  if (box.setAttribute) box.setAttribute('aria-modal', 'false');
  box.innerHTML = `<div class="promo-bar promo-exit-bar">
    <span class="promo-wait" id="promoHead">Come out where?</span>
    <span class="promo-exits">${ways.map((m, i) => {
      const said = exitWords(m);
      return `<button type="button" class="promo-pick promo-exit" data-move="${m.move}"
        data-exit="${m.to}" aria-keyshortcuts="${i + 1}" title="${esc(said.words)} (key ${
        i + 1})${esc(still)}"><span class="promo-arrow" aria-hidden="true">${
        said.arrow}</span>${esc(said.words)}</button>`;
    }).join('')}</span>
    <span class="promo-row"><span class="muted promo-esc">Esc: choose another move</span>
    <button type="button" class="quiet" id="promoCancel">Cancel</button></span>
  </div>`;
  box.querySelectorAll('[data-exit]').forEach(node => {
    const to = Number(node.dataset.exit);
    const moves = PROMO.filter(m => m.to === to);
    const light = () => lightExit(moves[0]);
    const dim = () => { if (LIT && LIT.to === to) lightExit(null); };
    node.onmouseenter = light;
    node.onfocus = light;
    node.onmouseleave = dim;
    node.onblur = dim;
    // A way out that is also a promotion asks what the piece becomes next.
    node.onclick = () => {
      closePromotion();
      if (moves.length === 1) send(moves[0].move); else choosePromotion(moves);
    };
  });
  document.getElementById('promoCancel').onclick = () => cancelPromotion();
  promotionInView(box, box.querySelector('.promo-bar'), box.querySelector('[data-exit]'));
}

// A route by its portal's number and square (Q838c): *through portal 2 on e3*, or with one
// pair *through the portal on e3*. Never by colour: the square already says which end.
function routeWords(m) {
  if (m.portal === undefined) return 'directly';
  const end = portalEnd(pieceAt(m.portal) ? pieceAt(m.portal).type : -1);
  const name = end ? portalName(end, portalPairs(SESSION.ruleset.portals)) : 'the portal';
  return `through ${name} on ${m.portalName}`;
}

// The piece-or-portal card (§5.3R): two tiles, the piece as the board draws it and the
// portal as the board draws it, each with the user's words and, where it is one move, the
// engine's text for it.
function whichCard(mover, word, still) {
  const W = SESSION.ruleset.width;
  const where = squareName(PROMO_WHICH.to % W, Math.floor(PROMO_WHICH.to / W));
  const head = 'Move the piece or the portal?';
  // A momentum end is a way in, not a square the piece stops on.
  const goesIn = PROMO_WHICH.piece.some(m => m.portal === PROMO_WHICH.to);
  const says = (PROMO_WHICH.portal
    ? `Your ${esc(word)} and the portal it stands in can both go to ${esc(where)}.`
    : `Your ${esc(word)} can ${goesIn ? 'go in by' : 'go to'} ${esc(where)}, and you may move the portal there.`)
    + ' Tap the one you want.' + still;
  const piece = mover ? { player: mover.player, type: mover.type, facing: mover.facing || 0 } : null;
  const at = foldOn(PROMO_WHICH.at) || portalOn(PROMO_WHICH.at);
  const end = at ? at.end : null;
  const tile = drawn => `<svg class="promo-piece" viewBox="0 0 100 100" aria-hidden="true">${drawn}</svg>`;
  const portalTile = !end ? '' : tile(isFold(end)
    ? foldGlyph(end, 0, 0, 100, { plain: true, numbered: true })
    : portalGlyph(end, 0, 0, 100, { numbered: true }));
  const one = moves => (moves && moves.length === 1 ? moves[0].text : '');
  const button = (which, drawn, words, text) => `<button type="button"
    class="promo-pick promo-route promo-which" data-which="${which}" aria-label="${esc(words)}${
    text ? `, ${esc(text)}` : ''}">${drawn}<span class="promo-way">${esc(words)}</span>${
    text ? `<span class="promo-text">${esc(text)}</span>` : ''}</button>`;
  const picks = `<div class="promo-pieces promo-routes">${
    button('piece', piece ? tile(drawPiece(piece, 0, 0, 100)) : '', 'Move the piece', one(PROMO_WHICH.piece))}${
    button('portal', portalTile, 'Move the portal',
           PROMO_WHICH.portal ? one(PROMO_WHICH.portal) : 'then choose where it goes')}</div>`;
  return { head, says, picks };
}

// ----- where a strike's survivor goes (Q769)
//
// A strike under `position: "choose"` takes the square and puts the piece it struck,
// changed by the strike, on an empty square of its own choosing: one move per landing
// square, and, when the striker promotes as it takes, per landing square *and* piece. That
// choice was the last row of small buttons under the board, naming the new piece by its
// letter (`b7 =Q`). It is the same card now, in two steps when there are two things to
// pick: first what the striker becomes, the pieces drawn large and named, exactly as
// above; then the square for the piece it took, each square a large target named as the
// board's edge names it, with *Change the piece* to go back a step. One step is skipped
// when it offers one answer only. *Look at the board*, Cancel, Escape, the keys and a pick
// being the move are the promotion's. The squares are the engine's moves' own
// (`landingName`), never squares the page works out.
function chooseLanding(candidates) {
  choosePromotion(candidates);
}

// The piece types a choice offers, in the engine's order, each once.
function promotionPieces(candidates) {
  const seen = [];
  for (const m of candidates) {
    if (m.promotion !== undefined && !seen.includes(m.promotion)) seen.push(m.promotion);
  }
  return seen;
}

// What the open choice asks now: `piece` (what the piece becomes) or `square` (where a
// strike's survivor goes, from the moves the piece picked leaves).
function promotionStage() {
  if (PROMO && PROMO_EXIT) return 'exit';
  if (PROMO && PROMO_WHICH) return 'which';
  if (PROMO && PROMO_ROUTE) return 'route';
  if (!PROMO || PROMO[0].landing === undefined) return 'piece';
  return promotionPieces(PROMO).length > 1 && PROMO_PIECE === null ? 'piece' : 'square';
}

function landingMoves() {
  return PROMO.filter(m => PROMO_PIECE === null || m.promotion === PROMO_PIECE);
}

// A piece picked where a square is still to choose: the move itself when only one square
// is left, else the squares.
function pickPiece(type) {
  const left = PROMO.filter(m => m.promotion === type);
  if (left.length === 1) { closePromotion(); send(left[0].move); return; }
  PROMO_PIECE = type;
  drawPromotion();
}

// The choice as it stands: the card over the dimmed board, or, while the reader looks at
// the board, the bar under it. Either way its first control takes the focus.
function drawPromotion() {
  const box = document.getElementById('promo');
  if (!box || !PROMO) return;
  if (PROMO_EXIT) { drawExits(box); return; }
  const candidates = PROMO;
  const mover = pieceAt(candidates[0].from);
  const word = promotionWord(mover);
  const stage = promotionStage();
  const landing = candidates[0].landing !== undefined;
  // The piece the strike took, as the reader sees it standing there now.
  const taken = esc(promotionWord(pieceAt(candidates[0].to)));
  box.hidden = false;
  box.onkeydown = promotionKeys;
  if (box.classList) box.classList.toggle('looking', PROMO_LOOKING);
  if (box.setAttribute) box.setAttribute('aria-modal', PROMO_LOOKING ? 'false' : 'true');
  if (PROMO_LOOKING) {
    box.innerHTML = `<div class="promo-bar">
      <span class="promo-wait" id="promoHead">${stage === 'which'
        ? 'Your move is waiting: the piece or the portal.'
        : stage === 'route'
        ? `Your ${esc(word)} is waiting for its way.`
        : stage === 'square'
        ? `The ${taken} you took is waiting to be placed.`
        : `Your ${esc(word)} is waiting to become a new piece.`}</span>
      <span class="promo-row"><button type="button" class="primary" id="promoBack">${
        stage === 'which' ? 'Choose which' : stage === 'route' ? 'Choose the way'
        : stage === 'square' ? 'Choose the square' : 'Choose the new piece'}</button>
      <button type="button" class="quiet" id="promoCancel">Cancel</button></span>
    </div>`;
    document.getElementById('promoBack').onclick = () => { PROMO_LOOKING = false; drawPromotion(); };
    document.getElementById('promoCancel').onclick = () => cancelPromotion();
    promotionInView(box, box.querySelector('.promo-bar'), document.getElementById('promoBack'));
    return;
  }
  const player = mover ? mover.player : SESSION.sideToMove;
  const names = SESSION.ruleset.pieceNames || [];
  const ids = SESSION.ruleset.pieceIds || [];
  const nameOf = type => names[type] || ids[type] || 'piece';
  const between = !!SESSION.correspondence;
  const still = between ? ' You can still change your move until you send it.' : '';
  let head, says, picks, back = '';
  if (stage === 'which') {
    ({ head, says, picks } = whichCard(mover, word, still));
  } else if (stage === 'route') {
    head = `Choose how your ${esc(word)} gets there`;
    says = 'This square can be reached more than one way. Tap the way you want.' + still;
    // Each way is the promotion's card: the piece drawn in the tile, as the promotion
    // draws the piece it becomes, then the way in words at the body's size (Q832), then
    // the move. The words never go in the tile, whose glyph type broke a long portal name
    // a letter or two a line out of the card.
    const piece = mover ? { player: mover.player, type: mover.type, facing: mover.facing || 0 } : null;
    picks = `<div class="promo-pieces promo-routes">${candidates.map(m =>
      `<button type="button" class="promo-pick promo-route" data-move="${m.move}"
        aria-label="${esc(routeWords(m))}, ${esc(m.text)}">${piece
        ? `<svg class="promo-piece" viewBox="0 0 100 100" aria-hidden="true">${
          drawPiece(piece, 0, 0, 100)}</svg>` : ''}<span class="promo-way">${
        esc(routeWords(m))}</span><span class="promo-text">${esc(m.text)}</span></button>`
    ).join('')}</div>`;
  } else if (stage === 'square') {
    const moves = landingMoves();
    head = `Place the ${taken} you took`;
    const picked = nameOf(PROMO_PIECE);
    says = (PROMO_PIECE !== null
      ? `Your ${esc(word)} becomes ${/^[aeiou]/i.test(picked) ? 'an' : 'a'} ${esc(picked)}. ` : '')
      + 'Tap a square.' + still;
    picks = `<div class="promo-pieces promo-squares">${moves.map(m =>
      `<button type="button" class="promo-pick promo-square" data-move="${m.move}"
        aria-label="square ${esc(m.landingName)}"><span class="promo-sq">${esc(m.landingName)}</span></button>`
    ).join('')}</div>`;
    if (PROMO_PIECE !== null) {
      back = '<button type="button" class="quiet" id="promoPiece">Change the piece</button>';
    }
  } else {
    head = `Choose what your ${esc(word)} becomes`;
    says = landing
      ? `Tap the piece you want. Then place the ${taken} you took.${still}`
      : between ? 'Tap a piece. You can still change your move until you send it.'
        : 'Tap the piece you want.';
    // A strike's pieces are its piece types, each once: the square comes next.
    const offered = landing
      ? promotionPieces(candidates).map(type => ({ type, attr: `data-piece="${type}"` }))
      : candidates.map(m => ({ type: m.promotion, attr: `data-move="${m.move}"` }));
    picks = `<div class="promo-pieces">${offered.map(({ type, attr }) => {
      const name = nameOf(type);
      const piece = { player, type, facing: mover ? mover.facing : 0 };
      return `<button type="button" class="promo-pick" ${attr}
        aria-label="${esc(name)}"><svg class="promo-piece" viewBox="0 0 100 100"
        aria-hidden="true">${drawPiece(piece, 0, 0, 100)}</svg><span>${esc(name)}</span></button>`;
    }).join('')}</div>`;
  }
  box.innerHTML = `<div class="promo-card">
    <h2 class="promo-head" id="promoHead">${head}</h2>
    <p class="promo-says">${says}</p>
    ${picks}
    <div class="promo-row">${back}<button type="button" class="quiet" id="promoLook"
      title="put this away to see the whole board; the move is not made yet">Look at the board</button>
    <button type="button" class="quiet" id="promoCancel">Cancel</button></div>
  </div>`;
  box.querySelectorAll('[data-move]').forEach(node => {
    node.onclick = () => { closePromotion(); send(Number(node.dataset.move)); };
  });
  box.querySelectorAll('[data-piece]').forEach(node => {
    node.onclick = () => pickPiece(Number(node.dataset.piece));
  });
  box.querySelectorAll('[data-which]').forEach(node => {
    node.onclick = () => pickWhich(node.dataset.which);
  });
  if (back) document.getElementById('promoPiece').onclick = () => { PROMO_PIECE = null; drawPromotion(); };
  document.getElementById('promoLook').onclick = () => { PROMO_LOOKING = true; drawPromotion(); };
  document.getElementById('promoCancel').onclick = () => cancelPromotion();
  promotionInView(box, box.querySelector('.promo-card'), box.querySelector('[data-move]')
    || box.querySelector('[data-piece]') || box.querySelector('[data-which]'));
}

// On a phone the board may be half off the screen when the pawn arrives: the board is
// scrolled until it is whole on the screen, then the part drawn (a card taller than a
// small board, the bar under it), neither where it is already in view, and its first
// control takes the focus without scrolling again.
function promotionInView(box, part, first) {
  for (const node of [box, part]) {
    if (node && typeof node.scrollIntoView === 'function') {
      node.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }
  }
  if (first && typeof first.focus === 'function') first.focus({ preventScroll: true });
}

// Put away with nothing moved and nothing picked; a move already held between two people
// stays held.
function cancelPromotion() {
  closePromotion();
  SELECTED = null;
  drawBoard();
}

function closePromotion() {
  PROMO = null;
  PROMO_LOOKING = false;
  PROMO_PIECE = null;
  PROMO_ROUTE = false;
  PROMO_EXIT = false;
  PROMO_WHICH = null;
  lightExit(null);
  const box = document.getElementById('promo');
  if (!box) return;
  box.hidden = true;
  box.innerHTML = '';
  if (box.classList) box.classList.toggle('looking', false);
}

// Whether the choice still belongs to the board on screen: the game as it stands, still
// going, and every move it offers still one the engine offers.
function promotionStands() {
  if (!PROMO || !SESSION || AT || SESSION.outcome) return false;
  return PROMO.every(m => (SESSION.moves || []).some(o => o.move === m.move && o.text === m.text));
}

// The choice's own keys, kept from the page's: an arrow here walks the pieces rather than
// stepping through the game, and Escape puts this choice away rather than a held move.
function promotionKeys(event) {
  const box = document.getElementById('promo');
  if (!box || !PROMO) return;
  const stops = [...box.querySelectorAll('button')];
  const picks = stops.filter(node => node.dataset.move !== undefined
    || node.dataset.piece !== undefined || node.dataset.which !== undefined);
  const at = stops.indexOf(document.activeElement);
  if (event.key === 'Escape') {
    event.preventDefault();
    event.stopPropagation();
    cancelPromotion();
  } else if (event.key === 'Tab') {
    // The focus stays inside the choice while it is open.
    event.preventDefault();
    event.stopPropagation();
    const next = (at + (event.shiftKey ? -1 : 1) + stops.length) % stops.length;
    if (stops.length) stops[at < 0 ? 0 : next].focus();
  } else if (event.key.indexOf('Arrow') === 0) {
    event.preventDefault();
    event.stopPropagation();
    if (!picks.length) return;
    const step = event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1;
    const here = picks.indexOf(document.activeElement);
    const next = here < 0 ? 0 : (here + step + picks.length) % picks.length;
    picks[next].focus();
  } else if (PROMO_EXIT && /^[1-9]$/.test(event.key)) {
    // The ways out of a portal by number, as the buttons are ordered (D6).
    event.preventDefault();
    event.stopPropagation();
    const pick = picks[Number(event.key) - 1];
    if (pick) pick.click();
  } else if (event.key === ' ' || event.key === 'Enter') {
    // The button's own press; the page's space (pause) never sees it.
    event.stopPropagation();
  }
}

// Arrow symbols for the four cardinal directions a piece can face.
const ROTATION_ARROWS = ['↑', '→', '↓', '←'];  // ↑ → ↓ ←
const ROTATION_LABELS = ['up', 'right', 'down', 'left'];

function chooseRotation(candidates) {
  const hand = document.getElementById('hand');
  const nf = SESSION.ruleset.nFacings || 4;
  const player = SESSION.sideToMove;
  const headings = (SESSION.ruleset.facingAngles || [])[player] || [];
  hand.innerHTML = '<span class="muted u-as-center">rotate to:</span>' +
    candidates.map(m => {
      const facing = m.rotation;
      // Map the facing angle to an arrow: 0°=up, 90°=right, 180°=down, 270°=left
      const angle = headings[facing] || 0;
      const arrowIdx = Math.round(angle / 90) % 4;
      const arrow = ROTATION_ARROWS[arrowIdx];
      const label = ROTATION_LABELS[arrowIdx];
      return `<button data-move="${m.move}" title="face ${label}" class="u-fs-1_5em u-minw-2_5em u-p-4-8">${arrow}</button>`;
    }).join('');
  hand.querySelectorAll('[data-move]').forEach(node => {
    node.onclick = () => send(Number(node.dataset.move));
  });
}

// A refused move is answered on the board's own control line, beside the status the
// reader is already watching, rather than at the top of a page they have scrolled past.
//
// A move whose answer never came back (Q659: the site restarting) may or may not have
// landed, and is never sent again (`api`). The page asks where the game is instead: a
// game that moved on is drawn as it is now, and one that did not is drawn as it was, with
// the selection cleared so the move can simply be made again.
function send(move) {
  SELECTED = null;
  DROP = null;
  INSPECT = null;
  if (SESSION.correspondence || SESSION.round) { (SESSION.round ? roundHold : corrHold)(move); return; }
  const id = SESSION.id, before = SESSION.history.length;
  const here = () => !!SESSION && SESSION.id === id;
  api('/api/session/' + id + '/move', { move }).then(s => applySnapshot(s))
    .catch(error => {
      // A refusal is a toast, never a box in the status strip, whose height is fixed (Q725).
      if (!error.unavailable) { fail(error); return; }
      api('/api/session/' + id).then(fresh => {
        if (!here()) return;
        if (fresh.history.length !== before) { applySnapshot(fresh); return; }
        drawBoard();
        fail(new Error(ACCOUNTS
          ? 'The site is updating or busy, so that move was not made. Try it again in a moment.'
          : 'The application did not answer, so that move was not made. Try it again.'));
      }).catch(() => {
        if (!here()) return;
        drawBoard();
        fail(error);
      });
    });
}

// ------------------------------------------------------------------ reviewing a game
//
// **A machine game is something to read, not only something to watch.** Both defects
// fixed on this surface were found by watching a game go past, which is a poor way to
// find anything: a viewer who sees something odd has no way back to it, and the game has
// moved on by the time they know they wanted to look. So the board keeps every position
// the game has been in, and a reader can stop it, walk back through them, and walk
// forward again.
//
// **The unit of a step is a beat, not a ply.** One submitted move can make two board
// changes -- the mover's, and then the wild-piece phase the rules run after it inside
// the same `make` -- and a stepper that went from ply to ply would step straight over
// the thing that was hardest to see in the first place. `plyBeats` already splits a ply
// into positions, and a beat's `board` is a real position rather than a frame of an
// animation, so the timeline is those positions, kept.
//
// **Kept, and never recomputed.** A step is an array index. Nothing here undoes a move,
// re-derives a position or works one out: every board in the list is a board the engine
// produced and the page was handed. That is what makes stepping back and then forward
// again show the same thing -- the two directions read one array in opposite orders --
// and it is also what keeps stepping on the right side of the rule this whole package
// lives inside, because reconstructing a position by undoing moves here would be the
// page answering a question that belongs to the engine.
//
// The cost is one board per beat: `width * height` small integers, so 64 numbers a beat
// on a chessboard and about 13,000 across a 200-ply game, plus the territory colouring
// and the hands where a ruleset has either. A game long enough for that to matter would
// take longer to play than a browser stays open.
let TIMELINE = [];         // every beat of this game, in order; [0] is the opening
let TIMELINE_FOR = null;   // which session it belongs to
let CURSOR = null;         // the beat on screen while reviewing; null while following
let AT = null;             // that beat itself -- what `position()` and `tint()` read
let PAUSED = false;        // the chain that plays a machine game is stopped

// One entry per beat. `plies` is how many moves had been made when this beat ended,
// which is the measure a take-back shortens the game in; `text` is the move the ply
// was, so the review line can say where you are in words rather than in an index.
// `walkers` is which piece took each of a phase's steps, as `plyBeats` found it, which
// is what lets the move list name the wolf as well as its squares.
// `coming` is where each wild piece was going next from this beat's board, and
// `comingThen` where after that, the core's own answers (`comingAt`), so a position
// stepped back to shows what was coming *then*.
function beatEntry(kind, board, steps, snapshot, plies, walkers, coming) {
  const move = plies > 0 ? snapshot.history[plies - 1] : null;
  return {
    kind: kind,
    board: board.slice(),
    steps: steps || null,
    walkers: walkers || null,
    coming: (coming && coming.coming) || [],
    comingThen: (coming && coming.comingThen) || [],
    lastMove: move || null,
    text: move ? move.text : '',
    plies: plies,
    sideToMove: snapshot.sideToMove,
    // The colouring and the hands as the ply left them. Neither is something a diff of
    // two boards could split between a ply's beats, and no rule that can express a wild
    // piece also claims ground or takes a piece in hand -- so the two beats of a ply
    // share the ply's own, and what differs between them is the board, which is the
    // whole of what the phase changes.
    colour: (snapshot.colour || []).slice(),
    hands: (snapshot.hands || []).map(hand => hand.slice()),
    // Schema 22 (Q843d): where the folds stood once this ply was made (the move's own
    // `folds`), else the snapshot's: a fold is no board byte, so a step back draws them here.
    folds: ((move && move.folds) || snapshot.folds || []).slice(),
    // Schema 30 (Q999c): the effects on the position the ply left (the move's own
    // `effects`), else the snapshot's: a step back draws the badges of its own position.
    effects: effectPairs((move && move.effects) || snapshot.effects),
    // Schema 29 (Q983b): the pieces resting on the position the ply left (the move's own
    // `resting`); nothing rests before the first move.
    resting: ((move && move.resting) || []).slice(),
  };
}

// Extend the timeline by whatever a new snapshot added, or shorten it by whatever a
// take-back removed.
//
// The position a ply started from is read off the *timeline*, not off what the board
// happens to be showing. A reply that lands while somebody is reading move six is still
// part of the game and has to be recorded from the position it was actually played in,
// and `SHOWN` at that moment is whatever the reader last looked at.
function record(snapshot) {
  const plies = snapshot.history.length;
  if (TIMELINE_FOR !== snapshot.id) { TIMELINE_FOR = snapshot.id; TIMELINE = []; }
  if (!TIMELINE.length) {
    TIMELINE = snapshot.replay
      ? replayBeats(snapshot)
      : [beatEntry('start', snapshot.board, null, snapshot, plies, null, snapshot)];
    return;
  }
  // A take-back is not a move: the plies after the position being restored are no
  // longer part of the game and their beats go with them.
  while (TIMELINE.length > 1 && TIMELINE[TIMELINE.length - 1].plies > plies) TIMELINE.pop();
  const tail = TIMELINE[TIMELINE.length - 1];
  if (tail.plies >= plies) return;         // a take-back, or a redraw of what is there
  // A round (schema 26) is two plies, both moves at once, split into beats as one ply is.
  const round = !!snapshot.round && plies - tail.plies === 2;
  if (plies - tail.plies > 1 && !round) {
    // More than one ply between two snapshots, which nothing on this page does today.
    // A diff across several moves does not split into beats, so it is recorded as one
    // position rather than as a guess at the ones in between.
    TIMELINE.push(beatEntry('ply', snapshot.board, null, snapshot, plies, null, snapshot));
    return;
  }
  const beats = plyBeats(snapshot.board, snapshot.lastMove);
  beats.forEach((beat, i) => {
    TIMELINE.push(beatEntry(beat.kind, beat.board, beat.steps, snapshot, plies, beat.walkers,
                            comingAt(beats, i, snapshot.lastMove)));
  });
}

// What was coming from one beat of a ply, as `{coming, comingThen}`, all of it the
// core's: from the board between the mover's move and the phase, the phase itself, which
// the core logged as it ran it, and then the first phase the core answered for the
// position the ply left; from the ply's last beat, the core's two answers for that
// position (`engine.coming.coming_phases`, carried on the move as `coming` and
// `comingThen`).
function comingAt(beats, i, move) {
  if (i < beats.length - 1) {
    return { coming: beats[i + 1].steps || [], comingThen: (move && move.coming) || [] };
  }
  return { coming: (move && move.coming) || [], comingThen: (move && move.comingThen) || [] };
}

// A session the server replayed arrives many plies in, with every position it passed
// through in `replay`. Each ply splits into beats exactly as a live one does.
function replayBeats(snapshot) {
  const at = ply => Object.assign({ history: snapshot.history }, snapshot.replay[ply]);
  const beats = [beatEntry('start', snapshot.replay[0].board, null, at(0), 0, null,
                           snapshot.replay[0])];
  for (let ply = 1; ply < snapshot.replay.length; ply++) {
    const move = snapshot.history[ply - 1];
    const split = plyBeats(snapshot.replay[ply].board, move);
    split.forEach((beat, i) => {
      beats.push(beatEntry(beat.kind, beat.board, beat.steps, at(ply), ply, beat.walkers,
                           comingAt(split, i, move)));
    });
  }
  return beats;
}

// Where the cursor is: the beat on screen, or the end of the game while following it.
function cursorAt() { return CURSOR === null ? TIMELINE.length - 1 : CURSOR; }

// Step by whole beats. Out of range does nothing at all -- pressing forward at the end
// of a running game must not stop it.
function stepBy(delta) {
  const to = cursorAt() + delta;
  if (to < 0 || to > TIMELINE.length - 1) return;
  showBeat(to);
}

// Show one beat. Reading a game and playing one are different activities, so stepping
// stops the machine: a stepper that a live game kept overwriting would not be a stepper.
//
// The last beat is a special case worth stating: it *is* the position the game is in,
// so standing on it is following the game rather than reviewing it. The status line and
// the verdict are true of it, a person may move in it, and the only thing still stopped
// is the chain -- which `resumeGame` starts again.
function showBeat(index) {
  if (TIMELINE.length < 2) return;
  const at = Math.max(0, Math.min(TIMELINE.length - 1, index));
  const live = at === TIMELINE.length - 1;
  const from = reviewTrip(cursorAt(), at);
  // A game somebody is playing has no pause button (B3 of `docs/archive/plan-ux.md`), so coming
  // back to the position the game is in is what starts it again. Without this, stepping
  // back in a game against the computer left it stopped with no control to restart it.
  if (live && SESSION.seats.includes('human')) return resumeGame(from);
  PAUSED = true;
  CURSOR = live ? null : at;
  AT = live ? null : TIMELINE[at];
  PHASE = TIMELINE[at].steps;
  SELECTED = null;
  DROP = null;
  drawBoard(from);
  drawPane();
}

// A step through the timeline that plays the move it walks over (Q836): exactly one beat
// forward, onto the mover's beat of a move through a portal. Returns the beat stepped from,
// marked `review`, for `drawBoard` to play the journey from; null for every other step,
// which hands the position over whole, as a step always has. A step back never plays a
// move backwards, for the reason a take-back does not: a move unmade is not a journey
// anybody took.
function reviewTrip(was, to) {
  if (to !== was + 1 || !TIMELINE[was] || !TIMELINE[to]) return null;
  const beat = TIMELINE[to];
  if (beat.kind !== 'move' || !beat.lastMove
      || (beat.lastMove.portal === undefined && beat.lastMove.foldIn === undefined)) return null;
  return { board: TIMELINE[was].board, review: true };
}

// Back to the game as it stands, and on with it. The only way out of review, and so the
// only place the machine chain is started again -- through `maybeAgent`, which is what
// checks the reply belongs to the game on screen and that there is still a board to draw
// it on. Resuming must not start a second game beside the one already running, and it
// cannot, because it does not start a chain of its own.
function resumeGame(from) {
  PAUSED = false;
  CURSOR = null;
  AT = null;
  PHASE = null;
  drawBoard(from || undefined);
  drawPane();
  maybeAgent();
}

// Stopping does not redraw the board: a game paused a moment after a move should finish
// showing that move, and a redraw would wipe it.
function pauseGame() {
  PAUSED = true;
  drawStatus();
  drawReview();
}

// Where in the game this beat is, in words. A step is a beat, so the count is in beats;
// what a reader recognises is the move, so the move is what it leads with.
function beatLine(entry, at) {
  const where = ' - step ' + (at + 1) + ' of ' + TIMELINE.length;
  if (entry.kind === 'start') return 'the opening position' + where;
  // Schema 26: a beat of a round names the round and both moves.
  const move = entry.lastMove;
  if (move && move.round !== undefined) {
    const first = (SESSION.history || [])[entry.plies - 2];
    const both = (first ? first.text + ' | ' : '') + move.text;
    return (entry.kind === 'phase' ? 'round ' + (move.round + 1) + ', ' + both
      + ', and then the wild pieces move' : 'after round ' + (move.round + 1) + ', ' + both)
      + where;
  }
  const number = (Math.floor((entry.plies - 1) / 2) + 1) + '.' + (entry.plies % 2 ? '' : '..');
  if (entry.kind === 'phase') {
    return number + ' ' + entry.text + ', and then the wild pieces move' + where;
  }
  return 'after ' + number + ' ' + entry.text + where;
}

// The controls, and the sentence saying where you are. Words rather than glyphs alone:
// this row is the first thing a reader meets when they want to go back and look at
// something again, and "|<" is not a word.
//
// Built once and then only *updated*. Stepping redraws the board, and a board redraw
// redraws this -- so a row rebuilt from a string would replace the button under the
// pointer on every click, which drops keyboard focus after each step and makes the one
// thing this row is for, pressing back several times, the thing it is worst at.
function drawReview() {
  const row = document.getElementById('review');
  if (!row || !SESSION) return;
  // Nothing has happened yet, so there is nothing to review and no row for it.
  if (TIMELINE.length < 2) { row.innerHTML = ''; row.dataset.built = ''; return; }
  if (row.dataset.built !== 'yes') {
    row.innerHTML = `
      <button id="atStart" title="the opening position">start</button>
      <button id="stepBack" title="one step back (left arrow)">&larr; back</button>
      <button id="pause"></button>
      <button id="stepOn" title="one step on (right arrow)">forward &rarr;</button>
      <button id="atEnd" title="the position as it stands">latest</button>
      <span class="muted small" id="where"></span>`;
    row.dataset.built = 'yes';
    document.getElementById('atStart').onclick = () => showBeat(0);
    document.getElementById('stepBack').onclick = () => stepBy(-1);
    document.getElementById('stepOn').onclick = () => stepBy(1);
    document.getElementById('atEnd').onclick = () => showBeat(TIMELINE.length - 1);
    document.getElementById('pause').onclick = () => { if (PAUSED) resumeGame(); else pauseGame(); };
  }
  const at = cursorAt();
  const first = at === 0, last = at === TIMELINE.length - 1;
  document.getElementById('atStart').disabled = first;
  document.getElementById('stepBack').disabled = first;
  document.getElementById('stepOn').disabled = last;
  document.getElementById('atEnd').disabled = last;
  const pause = document.getElementById('pause');
  // Pausing is only meaningful where the game goes on by itself with nobody to stop it,
  // which is a board nobody sits at. A game you are playing already waits for you, and
  // stepping back stops it; the way on is *latest*, which is four buttons rather than a
  // fifth that reads as a second kind of turn.
  pause.hidden = !!SESSION.outcome || SESSION.seats.includes('human');
  pause.textContent = PAUSED ? 'resume' : 'pause';
  pause.title = PAUSED
    ? 'go back to the game as it stands and play on'
    : 'stop before the next move';
  document.getElementById('where').textContent = beatLine(TIMELINE[at], at);
}

// Arrow keys step, shift and an arrow move the view across the seam, space stops and
// starts. Wired once for the life of the page rather than once per game, because
// `showBoard` runs again every time a board is opened and a listener per visit is a
// listener per visit forever. It does nothing unless a board is on screen, and nothing
// while somebody is typing into a field.
//
// The shift is on the modifier because the bare arrows already step through the game,
// and reading a game is the more common of the two.
function wireReviewKeys() {
  if (wireReviewKeys.done) return;
  wireReviewKeys.done = true;
  window.addEventListener('keydown', event => {
    if (!SESSION || !document.getElementById('review')) return;
    const tag = (event.target && event.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    // While the choice of a promotion is open, the keys are its own, wherever the focus
    // has wandered to (Q765).
    if (PROMO) { promotionKeys(event); return; }
    if (event.shiftKey && event.key.indexOf('Arrow') === 0) {
      const wrap = wrapAxes(SESSION.ruleset.wrap);
      const by = { ArrowLeft: [-1, 0], ArrowRight: [1, 0],
                   ArrowUp: [0, 1], ArrowDown: [0, -1] }[event.key];
      if (!by || (by[0] && !wrap.files) || (by[1] && !wrap.ranks)) return;
      event.preventDefault();
      shiftView(by[0], by[1]);
      return;
    }
    if (event.key === 'ArrowLeft') { event.preventDefault(); stepBy(-1); }
    else if (event.key === 'ArrowRight') { event.preventDefault(); stepBy(1); }
    else if (event.key === ' ') {
      // The key the pause button is, so it is offered exactly where that button is.
      if (SESSION.outcome || SESSION.seats.includes('human')) return;
      event.preventDefault();
      if (PAUSED) resumeGame(); else pauseGame();
    }
  });
}

// ------------------------------------------------------------- between two people
//
// Q570, `docs/archive/plan-return.md` §1.3 and §1.5. A game between two people is a match like
// any other, with three things the page does differently, all keyed on
// `snapshot.correspondence` (`site/correspondence.py::view`):
//
// **A picked move is held until Send.** The board offers the engine's legal moves as for
// any game; picking one puts it in `CORR_HELD`, the strip says *‹move› is not sent yet* and
// the slot under the board offers *Send your move* and *Change my move*. Until Send it can be
// changed on the board as well (Q740, `corrRepick`). Nothing reaches the server until Send,
// and a sent move is final. The board keeps the position as it stands until the move is sent, so what is on
// it is never a move the other person has not been told about.
//
// **Whose move it is, and the time left to move, are facts.** The strip says whose move
// it is, when the last move landed (*ana moved 3 hours ago*) and, in the server's words,
// how long the player to move has (*You have 4 days 6 hours to move.*, Q627: the user's
// dated exception to test 2, so a loss on time is never a surprise). The page never
// counts down: it shows the sentence it was handed, whole hours rounded down, in the
// strip's own plain text, with no ticking, no red and no emphasis that grows (§1.8's *no
// alarm red* outside resign). The rule of five days is said under the moves as a rule.
//
// **While it is the other person's move, the page asks for the match every 15 seconds,
// only while the tab is visible**, and never while it is yours (§1.5). One request at a
// time; a page that has moved on stops asking.
let CORR_HELD = null;          // the move picked and not sent yet, between two people
let CORR_POLL = null;  // the interval asking for the other person's move
const CORR_EVERY = 15000;

function stopCorrPoll() {
  if (CORR_POLL) { clearInterval(CORR_POLL); CORR_POLL = null; }
}

function corrTheirTurn() {
  const between = SESSION && SESSION.correspondence;
  return !!between && !SESSION.outcome && between.you !== null
    && between.you !== SESSION.sideToMove;
}

function pollCorr() {
  stopCorrPoll();
  if (!corrTheirTurn()) return;
  const id = SESSION.id;
  let asking = false;
  CORR_POLL = setInterval(() => {
    if (!SESSION || SESSION.id !== id || !document.getElementById('status')) {
      stopCorrPoll();
      return;
    }
    if (document.visibilityState !== 'visible' || asking || !corrTheirTurn()) return;
    asking = true;
    api('/api/session/' + id)
      .then(fresh => {
        asking = false;
        if (!fresh || !SESSION || SESSION.id !== id) return;
        if (corrChanged(SESSION, fresh)) corrApply(fresh);
      }).catch(() => { asking = false; });
  }, CORR_EVERY);
}

// In a game in rounds (Q918f) a seal that does not reveal changes no move list, so who has
// sealed is compared too: the booleans the reader is given, never the move.
function corrChanged(was, now) {
  const a = was.correspondence || {};
  const b = now.correspondence || {};
  const sealed = snapshot => String(((snapshot.round || {}).sealed) || '');
  return was.history.length !== now.history.length || !!was.outcome !== !!now.outcome
    || a.drawOffer !== b.drawOffer || a.mayCancel !== b.mayCancel
    || sealed(was) !== sealed(now);
}

// A new snapshot of a game between two people: the board as `applySnapshot` draws any
// move, then the controls redrawn (an offer, a cancel that is no longer possible), and the
// polling started or stopped by whose move it now is.
function corrApply(snapshot) {
  // A move still held survives an answer that did not move the game (a draw offered while
  // it waits); one the game has moved past is gone, here and in the browser's keeping.
  CORR_HELD = corrHeldKept(snapshot);
  applySnapshot(snapshot);
  refreshCorrControls();
  pollCorr();
}

function refreshCorrControls() {
  const between = SESSION.correspondence;
  const draw = document.getElementById('draw');
  const theirs = between.drawOffer !== null && between.drawOffer !== between.you;
  if (draw) {
    const mine = between.drawOffer !== null && between.drawOffer === between.you;
    sayOn(draw, theirs ? 'accept the draw' : mine ? 'draw offered' : 'offer a draw');
    draw.disabled = mine || !!SESSION.outcome;
  }
  // Q790: the other's standing offer can be said no to without a move.
  const decline = document.getElementById('declineDraw');
  if (decline) decline.hidden = !theirs || !!SESSION.outcome;
  const cancel = document.getElementById('cancel');
  if (cancel) cancel.hidden = !between.mayCancel || !!SESSION.outcome;
  // Q571. Once the game is over, *rematch* replaces the draw offer: a challenge to the
  // same person with the sides swapped, which waits for their answer like any other. A
  // game called off before it started has nothing to play again.
  const rematch = document.getElementById('rematch');
  const ended = !!SESSION.outcome && SESSION.outcome.reason !== 'cancelled';
  if (rematch) rematch.hidden = !ended;
  if (draw && SESSION.outcome) draw.hidden = true;
  drawEnded();
}

function wireCorr() {
  if (!SESSION || !SESSION.correspondence) return;
  wireHeldLeaving();
  const id = SESSION.id;
  const act =(action) => api('/api/session/' + id + '/' + action, {})
    .then(corrApply).catch(error => fail(error));
  const draw = document.getElementById('draw');
  if (draw) draw.onclick = () => {
    const between = SESSION.correspondence;
    const theirs = between.drawOffer !== null && between.drawOffer !== between.you;
    // One offer in any five of your own moves (Q790): the server's sentence, unasked.
    if (!theirs && between.mayOffer === false && between.offerWhy) {
      fail(new Error(between.offerWhy), draw);
      return;
    }
    act(theirs ? 'accept' : 'draw');
  };
  const decline = document.getElementById('declineDraw');
  if (decline) decline.onclick = () => act('decline');
  // Calling the game off asks once, like resign: it ends the game for both people.
  const cancel = document.getElementById('cancel');
  if (cancel) cancel.onclick = () => {
    if (cancel.dataset.armed !== 'yes') {
      cancel.dataset.armed = 'yes';
      sayOn(cancel, 'really cancel?');
      setTimeout(() => { cancel.dataset.armed = ''; sayOn(cancel, 'cancel the game'); }, ARM_HOLD);
      return;
    }
    act('cancel');
  };
  const rematch = document.getElementById('rematch');
  if (rematch) rematch.onclick = () => {
    rematch.disabled = true;
    api('/api/challenge', { rematch: id }).then(sent => {
      sayOn(rematch, 'rematch offered');
      document.getElementById('ground').textContent = `A rematch with the sides swapped `
        + `waits for ${sent.to} to accept or decline it, on your page under Challenges `
        + `you sent until then.`;
    }).catch(error => { rematch.disabled = false; fail(error, rematch); });
  };
  // Resign already goes through `/resign`; its answer redraws the controls here too.
  const resign = document.getElementById('resign');
  if (resign) {
    const pressed = resign.onclick;
    resign.onclick = () => {
      if (resign.dataset.armed === 'yes') {
        resign.dataset.armed = '';
        sayOn(resign, 'resign');
        act('resign');
        return;
      }
      pressed();
    };
  }
  refreshCorrControls();
}

function corrHold(move) {
  CORR_HELD = move;
  const entry = corrHeldEntry();
  if (entry) keepHeld(SESSION.id, SESSION.history.length, entry);
  drawStatus();
  drawBoard();
}

function corrHeldEntry() {
  return CORR_HELD === null ? null : (SESSION.moves || []).find(m => m.move === CORR_HELD) || null;
}

// The held move gone, here and in this browser's keeping (so the list of games waiting on
// you no longer says *Not sent yet*), without drawing anything.
function corrForget() {
  CORR_HELD = null;
  if (SESSION) forgetHeld(SESSION.id);
}

// Taking the held move back: nothing held, nothing picked, the board as the game stands.
function corrPutBack() {
  if (SESSION && SESSION.round) { roundPutBack(); return; }
  corrForget();
  SELECTED = null;
  DROP = null;
  NOTE = '';
  drawStatus();
  drawBoard();
}

// **A held move can be changed on the board until it is sent** (Q740; the user could not
// see how to change a move, and never found Q733's *put it back* link). A tap on the held move itself --
// its piece, the square it lands on, the faded piece standing there -- takes it back, and
// the answer is true: the tap is spent. Otherwise the held piece counts as the one picked,
// so another of its squares (the dots still drawn round it) replaces the move at once,
// through the ordinary pick in `clickSquare`; another piece of yours that can move is
// picked instead of it there; and a tap anywhere else leaves the held move as it is.
function corrRepick(square) {
  const held = corrHeldDrawn();
  if (!held || DROP !== null) return false;
  if (square === held.from || square === held.to || square === held.dest) {
    corrPutBack();
    return true;
  }
  if (SELECTED === null && held.from !== null) SELECTED = held.from;
  return false;
}

function wireCorrHeld() {
  const go = document.getElementById('sendHeld');
  const change = document.getElementById('changeHeld');
  if (change) change.onclick = corrPutBack;
  if (go) go.onclick = () => {
    const move = CORR_HELD, id = SESSION.id;
    if (move === null) return;
    go.disabled = true;
    if (change) change.disabled = true;
    sayOn(go, 'Sending…');
    // Once it is on its way the move is no longer the reader's to change: the board takes
    // no tap until the answer comes (`clickSquare`).
    CORR_SENDING = true;
    api('/api/session/' + id + '/move', { move })
      .then(fresh => { CORR_SENDING = false; forgetHeld(id); corrApply(fresh); })
      .catch(error => { CORR_SENDING = false; corrPutBack(); fail(error); });
  };
  wireHeldKeys();
}

let CORR_SENDING = false;  // a held move on its way to the server

// Escape takes a held move back too, for a reader at a keyboard (Q740); the two buttons are
// reached by Tab like any other. Once per page, like `wireReviewKeys`, and nothing while
// somebody is typing into a field or the move is on its way.
function wireHeldKeys() {
  if (wireHeldKeys.done || typeof window.addEventListener !== 'function') return;
  wireHeldKeys.done = true;
  window.addEventListener('keydown', event => {
    // An Escape the promotion's choice took (Q765) is not also a take-back.
    if (event.key !== 'Escape' || event.defaultPrevented || PROMO) return;
    if (CORR_HELD === null || CORR_SENDING) return;
    if (!SESSION || !SESSION.correspondence || !document.getElementById('corrSend')) return;
    const tag = (event.target && event.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    event.preventDefault();
    corrPutBack();
  });
}

// **The held move is kept in this browser** (Q733), so leaving the page and coming back
// finds it still on the board, still unsent: `localStorage`, one entry per game, holding
// how many moves the game had when it was picked and the move's number and words. It is
// given back only to the same game at the same move, and only while the engine still
// offers that move under those words; anything else is forgotten. It never reaches the
// server, so the other player never sees it, and nothing here says they have.
const HELD_KEY = 'chess_variant_sandbox.held.';

function keepHeld(id, ply, entry) {
  try {
    window.localStorage.setItem(HELD_KEY + id,
      JSON.stringify({ ply: ply, move: entry.move, text: entry.text }));
  } catch (error) { /* not kept: the move is still held on this page */ }
}

function forgetHeld(id) {
  try { window.localStorage.removeItem(HELD_KEY + id); } catch (error) { /* nothing kept */ }
}

function keptHeld(id) {
  try {
    const kept = JSON.parse(window.localStorage.getItem(HELD_KEY + id) || 'null');
    return kept && typeof kept === 'object' ? kept : null;
  } catch (error) { return null; }
}

// Whether this browser holds an unsent move for a game at `plies` moves: what the list of
// games waiting on you reads (`waitingHtml`). True only of the game as it still stands.
function heldWaiting(id, plies) {
  const kept = keptHeld(id);
  return !!kept && kept.ply === plies;
}

// The move to hold on a snapshot of a game between two people: the one kept for it, if
// the game is where it was when the move was picked and it is still the reader's move.
function corrHeldKept(snapshot) {
  const between = snapshot && snapshot.correspondence;
  if (!between) return null;
  const kept = keptHeld(snapshot.id);
  if (!kept) return null;
  const mine = !snapshot.outcome && between.you !== null && between.you === snapshot.sideToMove;
  const offered = (snapshot.moves || []).some(m => m.move === kept.move && m.text === kept.text);
  if (mine && kept.ply === (snapshot.history || []).length && offered) return kept.move;
  forgetHeld(snapshot.id);
  return null;
}

// Leaving the page with a move picked and not sent asks first, in the browser's own words
// (a page may not choose them). The move is kept either way.
let HELD_LEAVING = false;
function wireHeldLeaving() {
  if (HELD_LEAVING || typeof window.addEventListener !== 'function') return;
  HELD_LEAVING = true;
  window.addEventListener('beforeunload', event => {
    if (CORR_HELD === null || !SESSION || !SESSION.correspondence) return;
    event.preventDefault();
    event.returnValue = '';
  });
}

// ----- the held move on the board
//
// Drawn so it cannot be taken for a move that was made (Q733): the squares it leaves and
// lands on are outlined in dashes, an arrow runs between them in dashes, and the piece
// stands on the square it would land on faded, while the pieces on both squares stay
// where they are, faded too. A sent move is the other kind of mark entirely, the solid
// tint of the last move, which stays on the other player's last move while this one is
// held, so the two read as two things. `--held` is dark and edged in `--held-halo`, so
// it reads on both woods in both themes (`tests/test_style.py`).
const HELD_FADE = 0.45;

// The held move's squares: where it leaves (null for a piece put down from the hand), the
// square it names, where the piece comes to rest (a strike's landing, else that square),
// and the piece as it would stand there -- its type after a promotion, its facing after a
// turn. Nothing here works out what the move does to anything else on the board.
//
// In a game in rounds by correspondence (Q918f) the reader's own sealed move, which the
// server gives back to its seat alone (`round.yours`), is drawn the same way while it waits
// for the other's: it is the reader's move, not one made, and nothing else on the board
// says what it is.
function corrHeldDrawn() {
  if (!SESSION || !(SESSION.correspondence || SESSION.round)) return null;
  const sealed = SESSION.round && SESSION.correspondence ? SESSION.round.yours || null : null;
  const entry = corrHeldEntry() || sealed;
  if (!entry) return null;
  const from = entry.from === null || entry.from === undefined ? null : entry.from;
  const was = from === null ? null : pieceAt(from);
  const mover = sealed && entry === sealed ? SESSION.correspondence.you : SESSION.sideToMove;
  const piece = entry.drop !== undefined
    ? { player: mover, type: entry.drop, facing: 0 }
    : was ? Object.assign({}, was) : null;
  if (piece && entry.promotion !== undefined) piece.type = entry.promotion;
  if (piece && entry.rotation !== undefined) piece.facing = entry.rotation;
  const dest = entry.landing !== undefined && entry.landing !== null ? entry.landing : entry.to;
  const held = { from: from, to: entry.to, dest: dest, piece: piece, text: entry.text };
  // A move through a portal: the end it goes in by and the one it comes out beside (the
  // server's `portal_transit`), so the arrow is drawn in its two legs.
  if (entry.portal !== undefined && entry.portal !== null) {
    held.portal = entry.portal;
    held.portalOut = entry.portalOut;
  }
  return held;
}

function heldOutline(x, y, cell, arrived) {
  const dash = `stroke-dasharray="${cell * 0.16} ${cell * 0.1}"`;
  return cellShape(x, y, cell, FRAME, `fill="none" stroke="var(--held-halo)"
           stroke-width="${cell * 0.11}" opacity="0.9" pointer-events="none"`, 0.07)
    + cellShape(x, y, cell, FRAME, `fill="none" stroke="var(--held)"
           stroke-width="${cell * 0.06}" ${dash} pointer-events="none"
           data-held="${arrived ? 'to' : 'from'}"`, 0.07);
}

function heldHtml(held, cell, W, H, wrap) {
  let out = '';
  if (held.from !== null && held.from !== held.dest && held.piece) {
    const allow = wrapAllow(wrap, held.piece.player);
    // Q772d: a move through a portal is held as it would be made, in its two legs (§4's
    // trail, `portalLegsHtml`): into the end it goes in by, and out of the other end to
    // where it lands. One arrow from where it leaves to where it lands crossed the board
    // through whatever lay between, a wall included.
    const legs = held.portal !== undefined && held.portalOut !== undefined
      ? [[held.from, held.portal], [held.portalOut, held.dest]] : [[held.from, held.dest]];
    const drawn = legs.map(([from, to]) => heldArrowHtml(from, to, cell, W, H, allow)).join('');
    if (drawn) out += `<g data-held="arrow" pointer-events="none" stroke-linecap="round"
          stroke-linejoin="round">${drawn}</g>`;
  }
  if (held.piece) {
    const at = squareXY(held.dest, cell, FRAME);
    out += `<g data-held="piece" opacity="0.7">${drawPiece(held.piece, at[0], at[1], cell)}</g>`;
  }
  return out;
}

// One dashed arrow of a held move, from one square to another along the mover's own step
// (`wrapStep`, its seams), or nothing for a step too short to carry one. `view` is the
// board's (the reveal of an ordered round draws a lost move's arrow on the animated view).
function heldArrowHtml(from, to, cell, W, H, allow, view = FRAME) {
  const f = n => n.toFixed(3);
  const at = squareXY(from, cell, view);
  const step = stepXY(wrapStep(from, to, W, H, allow), cell, view);
  const length = Math.hypot(step[0], step[1]);
  if (!(length > cell * 0.6)) return '';
  const ux = step[0] / length, uy = step[1] / length;
  const x1 = at[0] + cell / 2 + ux * cell * 0.3, y1 = at[1] + cell / 2 + uy * cell * 0.3;
  const x2 = at[0] + cell / 2 + ux * (length - cell * 0.34);
  const y2 = at[1] + cell / 2 + uy * (length - cell * 0.34);
  const head = cell * 0.2;
  const back = [x2 - ux * head, y2 - uy * head];
  const wing = [-uy * head * 0.75, ux * head * 0.75];
  const tip = `M${f(back[0] + wing[0])} ${f(back[1] + wing[1])}L${f(x2)} ${f(y2)}L${
    f(back[0] - wing[0])} ${f(back[1] - wing[1])}Z`;
  const line = `x1="${f(x1)}" y1="${f(y1)}" x2="${f(back[0])}" y2="${f(back[1])}"`;
  return `<line ${line} stroke="var(--held-halo)"
      stroke-width="${f(cell * 0.15)}" opacity="0.85"/><path d="${tip}"
      fill="var(--held-halo)" stroke="var(--held-halo)" stroke-width="${f(cell * 0.09)}"
      opacity="0.85"/><line ${line} stroke="var(--held)" stroke-width="${f(cell * 0.075)}"
      stroke-dasharray="${f(cell * 0.16)} ${f(cell * 0.11)}"/><path d="${tip}"
      fill="var(--held)" stroke="var(--held)" stroke-width="${f(cell * 0.03)}"/>`;
}

// ----- the slot under the board
//
// What a person does next in a game between two people, in the largest words on the
// frame, kept the same height in every state so the board never moves (Q725's rule for
// the strip, here for the slot; on a phone its last words, `.more`, go): with a move
// held, **Send your move** as the frame's one
// big button, *Change my move* beside it (Q740), and that the other player has not seen it; on your
// move with nothing held, how to pick one and that nothing goes until it is sent; on
// theirs, that the move went and the page can be closed.
// Nothing for somebody watching, and nothing once the game is over.
function drawCorrSend() {
  const slot = document.getElementById('corrSend');
  if (!slot || !SESSION) return;
  const between = SESSION.correspondence;
  // A game in rounds, by correspondence too (Q918f): the seal's slot, not the send's. A spy
  // game's (schema 27) says its phase while it is played by turns, at one screen; between
  // two people a turn is sent as any move is, and the strip says the phase (Q935e).
  if (SESSION.round) { drawRoundSend(slot); return; }
  if (SESSION.spy && !between) { drawSpySend(slot); return; }
  if (!between || between.you === null || SESSION.outcome) {
    slot.hidden = true;
    slot.innerHTML = '';
    return;
  }
  slot.hidden = false;
  const other = esc(between.players[1 - between.you]);
  const held = corrHeldEntry();
  let says, row = '';
  if (held) {
    // Q740: *Change my move* is a plain button as tall as the one that sends, where *put
    // it back* was a link nobody found, and the words say the board takes a change too.
    // No name here: a long one pushed how to change it out of the slot's two lines.
    says = `<b>Not sent yet.</b> To change it, tap another square or press <b>Change my
      move</b>.`;
    row = `<button id="sendHeld" class="primary send-big">Send your move</button>
      <button id="changeHeld" class="send-change">Change my move</button>`;
  } else if (between.you === SESSION.sideToMove) {
    says = `<b>Your move.</b> Pick it on the board, then press <b>Send your move</b>.<span
      class="more"> Nothing goes to ${other} before that.</span>`;
  } else {
    const last = SESSION.history.length ? SESSION.history[SESSION.history.length - 1] : null;
    says = last && last.player === between.you
      ? `<b>Your move is sent.</b> It is ${other}'s turn now.<span class="more"> You can
        close this page.</span>`
      : `<b>It is ${other}'s turn.</b><span class="more"> You can close this page; the game
        waits for you.</span>`;
  }
  slot.classList.toggle('held', !!held);
  slot.innerHTML = `<div class="send-says">${says}</div>${row ? `<div class="send-row">${row}</div>` : ''}`;
  wireCorrHeld();
}

// When something happened, in words: a fact about the past, never time left.
function corrAgoWords(stamp) {
  if (!stamp) return '';
  const seconds = Math.max(0, (Date.now() - Date.parse(stamp)) / 1000);
  if (seconds < 60) return 'just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? '' : 's'} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? '' : 's'} ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? '' : 's'} ago`;
}

function corrStatus() {
  // A game in rounds has no player to move: the round's own strip (Q918f).
  if (SESSION.round) return roundStatusHtml();
  const between = SESSION.correspondence;
  const mover = SESSION.sideToMove;
  const check = SESSION.inCheck ? checkPill() : '';
  const last = SESSION.history.length ? SESSION.history[SESSION.history.length - 1] : null;
  const when = (between.times || [])[SESSION.history.length - 1];
  const fact = last
    ? `${esc(between.players[last.player])} moved ${esc(last.text)}${when ? ', ' + esc(corrAgoWords(when)) : ''}.`
    : `Nobody has moved yet.`;
  const offer = between.drawOffer === null ? ''
    : between.drawOffer === between.you ? ' You have offered a draw.'
    : ` ${esc(between.players[between.drawOffer])} offers a draw.`;
  const held = corrHeldEntry();
  // The strip says it in words; the button that sends it is the slot's, under the board
  // (`drawCorrSend`), so there is one of it. It says the move can still be changed (Q740),
  // since on a phone the slot is its two buttons alone.
  if (held) {
    return `<b>${esc(held.text)}</b> is not sent yet. You can change it until you send it.${
      check}`;
  }
  const left = between.toMove ? ` ${esc(between.toMove)}` : '';
  // A spy game's turn says its phase (Q935e): both spies standing, or the reply.
  const spy = corrSpyNoteHtml();
  if (between.you === mover) {
    return `<b>Your move.</b> ${fact}${left}${offer}${check}${
      NOTE ? `<span class="frame-note">${esc(NOTE)}</span>` : ''}${spy}`;
  }
  if (between.you === null) {
    return `${esc(between.players[mover])} to move. ${fact}${left}${check}${spy}`;
  }
  return `${esc(between.players[mover])}'s move. ${fact}${left}${offer}${check}${spy}`;
}

// The day beside each move of a game between two people (s4 of the mock): when it
// landed, as a date. Nothing for any other game, which is played in one sitting.
function corrMovedDayHtml(index) {
  const between = SESSION && SESSION.correspondence;
  const when = between && (between.times || [])[index];
  if (!when) return '';
  const day = new Date(Date.parse(when));
  const said = day.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  return ` <span class="muted small" title="${esc(String(when))}">${esc(said)}</span>`;
}

function corrRulesHtml() {
  const between = SESSION.correspondence;
  return `<ul class="rules muted small u-mt-10">${
    (between.rules || []).map(rule => `<li>${esc(rule)}</li>`).join('')}</ul>`;
}
