// A game played in rounds, on the play board (schema 26, Q918c, `docs/plan-simultaneous.md`
// §6.6). Loaded after `board.js`, whose board, strip and slot it fills in where
// `SESSION.round` says the game has no side to move.
//
// **Choosing.** A move is picked on the board as in any game, and then held: drawn as the
// held move between two people is (`heldHtml`), the piece faded on the square it would end
// on, until the person seals it with *Seal my move* (or Enter) or changes it (a tap on
// another square, *Change my move*, or Escape). Nothing reaches the server before the seal.
//
// **The computer commits blind.** Its move is sealed on the round's board as the round
// opens, by the core's own round player, and never sent to the page before the reveal: a
// snapshot says only that it is in (`round.sealed`). `web/play.py` seals it before it
// accepts the person's, whatever the page does.
//
// **The reveal.** Both moves are drawn at once (`animateRound`), from the board before the
// round, the board after it and the core's account of what happened in between
// (`lastMove.reveal`): a collision draws the two pieces arriving together and the loser, or
// both, broken there; a swap draws the two passing each other; a void move steps toward
// its target and comes back. The round's sentences are the server's (`round_reveal`), in
// the slot under the board and in the move list.
//
// **The standoff.** Two pieces that could only take each other by a capture-only move, after
// one void round (the core's `voidRun`) or two in a row (`bars`), are linked on the board
// (`roundMarks`, `standoffLine`), so a person knows why the capture is not offered.
//
// Nothing here works out a move, an outcome or a choice. Who won a collision, which move
// was void and what is barred are the core's; the page draws them.

let ROUND_SENDING = false;   // a sealed move on its way to the server

// ----- the pure parts, held by tests in node

// The standoff marks of a round, from the core's state as the snapshot carries it: the
// void run's pair, and every barred pair. `[{kind, squares}]`, player 0's square first.
function roundMarks(round) {
  if (!round) return [];
  const marks = [];
  if (round.voidRun) marks.push({ kind: 'voidRun', squares: round.voidRun.slice() });
  for (const bar of round.bars || []) marks.push({ kind: 'barred', squares: bar.slice() });
  return marks;
}

// What a revealed round shows, as motions: each seat's move as the core reported it, and
// what each one came to. `before` and `after` are the boards either side of the round
// (before any wild piece's walk), `reveal` the round's `lastMove.reveal`.
//
// Each move is `{player, from, to, kind}`: `arrive` (it ended where it was going, whatever
// stood there or had left), `win` and `lose` (one of two pieces that met on one square;
// both `lose` where neither survived), `void` (it went for a piece that had moved, and
// stayed), `drop` (a piece put down from the hand), `turn` (a piece turned where it stands,
// or a move that goes nowhere). A void piece that was then taken where it stood is
// `taken: true`. `taken` lists the other squares a piece was taken off -- a piece standing
// where a move arrived, a blast's -- read off the two boards, since a piece missing after
// the round is a fact of the positions, not a choice; `meeting` is the square two pieces
// met on, with its winner (`-1` for none), or null.
function roundStory(before, after, reveal) {
  const events = (reveal && reveal.events) || [];
  const collision = events.find(event => event.kind === 'collision') || null;
  const meeting = collision ? { square: collision.square, winner: collision.player } : null;
  const moves = ((reveal && reveal.moves) || []).map(move => {
    const out = { player: move.player, from: move.from, to: move.to };
    if (move.from === null || move.from === undefined) {
      out.from = null;
      out.kind = meeting && meeting.square === move.to
        ? (meeting.winner === move.player ? 'win' : 'lose') : 'drop';
    } else if (move.void) {
      out.kind = 'void';
      out.taken = before[move.from] !== after[move.from];
    } else if (move.from === move.to) {
      out.kind = 'turn';
    } else if (meeting && meeting.square === move.to) {
      out.kind = meeting.winner === move.player ? 'win' : 'lose';
    } else {
      out.kind = 'arrive';
    }
    return out;
  });
  // A square a moving piece left is not a capture there, nor is the meeting square; a void
  // piece taken where it stood is its own move's story.
  const left = new Set(moves.filter(m => m.kind !== 'void' && m.kind !== 'turn'
                                         && m.from !== null).map(m => m.from));
  const voided = new Set(moves.filter(m => m.kind === 'void').map(m => m.from));
  const taken = [];
  for (let square = 0; square < before.length; square++) {
    if (!before[square] || before[square] === after[square]) continue;
    if (left.has(square) || voided.has(square)) continue;
    if (meeting && meeting.square === square) continue;
    if (moves.some(m => m.kind === 'turn' && m.from === square && after[square])) continue;
    taken.push(square);
  }
  return { moves: moves, taken: taken, meeting: meeting };
}

// One row of the move list for a round: its number, both moves in seat order with what
// each came to, and the round's sentences under them.
function roundRowHtml(number, first, second) {
  const said = move => {
    const result = move && move.result;
    if (!result) return '';
    const mark = result === 'void' ? 'void' : result === 'won' ? 'won'
      : result === 'lost' ? '✕ lost' : '✕ both lost';
    return ` <span class="muted small" data-result="${esc(result)}">${esc(mark)}</span>`;
  };
  const words = ((second && second.reveal && second.reveal.words) || [])
    .map(line => `<div class="muted small">${esc(line)}</div>`).join('');
  return `${number}. ${esc(first.text)}${said(first)}${lookedHtml(first)} <span
    class="muted">|</span> ${esc(second.text)}${said(second)}${lookedHtml(second)}${words}`;
}

// ----- the strip and the slot

// The strip's words while a round is open. `SESSION.round.choosing` is the person the board
// offers moves to; the other seat is the computer, or the other person at this screen.
function roundStatusHtml() {
  const round = SESSION.round;
  const number = `<span class="muted">Round ${round.number}.</span> `;
  const held = corrHeldEntry();
  if (ROUND_SENDING) {
    return `${number}<b>Your move is in:</b> waiting for the reveal.`;
  }
  if (held) {
    return `${number}<b>${esc(held.text)}</b> is your choice, not sealed yet. You can change
      it until you seal it.`;
  }
  if (SESSION.correspondence) return roundCorrStatusHtml(number);
  const seat = round.choosing;
  if (seat === null || seat === undefined) return number;
  const other = 1 - seat;
  // Two people at one screen are both *you*: each is named by their side alone.
  const both = SESSION.seats[other] === 'human';
  const who = player => esc(sideName(player)) + (both ? '' : ` (${esc(seatLabel(player))})`);
  const theirs = round.sealed[other]
    ? `${who(other)} has sealed a move, unseen.${both ? ` Pass the screen to ${
      esc(sideName(seat))}.` : ''}` : '';
  const note = NOTE ? `<span class="frame-note">${esc(NOTE)}</span>` : '';
  return `${number}${seatDot(seat)}${who(seat)} to choose. ${theirs}${note}`;
}

// The strip in a game in rounds between two people at two screens (Q918f): the round, the
// reader's own sealed move if it is in (the server gives it to its own seat alone), whether
// the other's is in (a boolean: never which), when the last round was revealed, the
// server's sentence of the time left, and a draw offer. Nothing here is worked out: the
// round's state is the snapshot's and the sentences are the server's.
function roundCorrStatusHtml(number) {
  const round = SESSION.round;
  const between = SESSION.correspondence;
  const you = between.you;
  const left = between.toMove ? ` ${esc(between.toMove)}` : '';
  if (you === null || you === undefined) {
    return `${number}${esc(between.players[0])} and ${esc(between.players[1])} choose.${left}`;
  }
  const other = esc(between.players[1 - you]);
  const offer = between.drawOffer === null || between.drawOffer === undefined ? ''
    : between.drawOffer === you ? ' You have offered a draw.' : ` ${other} offers a draw.`;
  const when = (between.times || [])[SESSION.history.length - 1];
  const fact = SESSION.history.length
    ? ` Round ${round.number - 1} was revealed${when ? ' ' + esc(corrAgoWords(when)) : ''}.`
    : '';
  if (round.yours) {
    return `${number}<b>Your move is sealed:</b> ${esc(round.yours.text)}. ${other} has not
      sealed theirs yet.${fact}${left}${offer}`;
  }
  const theirs = round.sealed[1 - you] ? ` ${other} has sealed a move, unseen.` : '';
  const note = NOTE ? `<span class="frame-note">${esc(NOTE)}</span>` : '';
  return `${number}<b>Your move.</b>${theirs}${fact}${left}${offer}${note}`;
}

// The slot under the board in a game in rounds with a person in it: *Seal my move* and
// *Change my move* while a move is held; else what to do, and what the last round did.
// Between two people at two screens (Q918f), a sealed move waits for the other's: the
// slot says so, and that the page can be closed.
function drawRoundSend(slot) {
  if (SESSION.outcome || AT) {
    const words = lastRoundWords();
    slot.hidden = !words;
    slot.classList.toggle('held', false);
    slot.innerHTML = words ? `<div class="send-says">${words}</div>` : '';
    return;
  }
  const between = SESSION.correspondence || null;
  if (between && (between.you === null || between.you === undefined)) {
    slot.hidden = true;
    slot.textContent = '';
    return;
  }
  slot.hidden = false;
  const held = corrHeldEntry();
  let says, row = '';
  if (ROUND_SENDING) {
    says = '<b>Your move is in.</b> Waiting for the reveal.';
  } else if (held) {
    says = `<b>Not sealed yet.</b> To change it, tap another square or press <b>Change my
      move</b>.<span class="more"> Both moves are revealed together, once both are in.</span>`;
    row = `<button id="sealHeld" class="primary send-big" title="seal it (Enter)">Seal my
      move</button><button id="changeHeld" class="send-change" title="put it back (Esc)"
      >Change my move</button>`;
  } else if (between && between.you !== null && SESSION.round.yours) {
    const other = esc(between.players[1 - between.you]);
    says = `<b>Your move is sealed.</b> It is revealed with ${other}'s, once theirs is
      in.<span class="more"> You can close this page.</span>`;
  } else {
    const words = lastRoundWords();
    says = words || `<b>Choose your move</b> on the board, then press <b>Seal my move</b>.<span
      class="more"> Neither player sees the other's move until both are in; then both are
      made at once.</span>`;
  }
  slot.classList.toggle('held', !!held && !ROUND_SENDING);
  slot.innerHTML = `<div class="send-says">${says}</div>${row ? `<div class="send-row">${row}</div>` : ''}`;
  const seal = document.getElementById('sealHeld');
  const change = document.getElementById('changeHeld');
  if (change) change.onclick = roundPutBack;
  if (seal) seal.onclick = roundSeal;
  wireRoundKeys();
}

// The last revealed round in the server's sentences, or nothing for a round with nothing
// to tell (two moves that each did what they said).
function lastRoundWords() {
  const last = AT ? AT.lastMove : SESSION.lastMove;
  const words = (last && last.reveal && last.reveal.words) || [];
  if (!words.length) return '';
  return `<b>Last round:</b> ${words.map(esc).join(' ')}`;
}

// ----- choosing and sealing

// A move picked on the board, held until it is sealed. Between two people at two screens
// it is kept in this browser as an unsent move is (Q733, `keepHeld`), so leaving the page
// and coming back finds it still held, still unsealed.
function roundHold(move) {
  CORR_HELD = move;
  const entry = SESSION.correspondence ? corrHeldEntry() : null;
  if (entry) keepHeld(SESSION.id, SESSION.history.length, entry);
  drawStatus();
  drawBoard();
}

function roundPutBack() {
  if (ROUND_SENDING) return;
  if (SESSION && SESSION.correspondence) forgetHeld(SESSION.id);
  CORR_HELD = null;
  SELECTED = null;
  DROP = null;
  NOTE = '';
  drawStatus();
  drawBoard();
}

// The held move, sealed. Against the computer the round is revealed in the same answer,
// its move having been sealed as the round opened; between two people at one screen the
// answer asks the other for theirs. Between two people at two screens (Q918f) the answer
// is the reader's: their move sealed and waiting, or the round revealed if the other's was
// in, and it goes through `corrApply`, which redraws the offer and cancel and starts the
// asking for the other's move.
function roundSeal() {
  const move = CORR_HELD;
  if (move === null || ROUND_SENDING || !SESSION) return;
  const id = SESSION.id;
  ROUND_SENDING = true;
  drawStatus();
  api('/api/session/' + id + '/move', { move })
    .then(fresh => {
      ROUND_SENDING = false;
      if (!SESSION || SESSION.id !== id) return;
      CORR_HELD = null;
      if (SESSION.correspondence) {
        forgetHeld(id);
        corrApply(fresh);
        return;
      }
      applySnapshot(fresh);
    })
    .catch(error => {
      ROUND_SENDING = false;
      if (!SESSION || SESSION.id !== id) return;
      roundPutBack();
      fail(error);
    });
}

// Enter seals a held move and Escape puts it back, for a reader at a keyboard; the two
// buttons are reached by Tab like any other. Once per page, and nothing while somebody is
// typing into a field, a button has the key, or the promotion's choice is open.
function wireRoundKeys() {
  if (wireRoundKeys.done || typeof window.addEventListener !== 'function') return;
  wireRoundKeys.done = true;
  window.addEventListener('keydown', event => {
    if (event.defaultPrevented || PROMO || !SESSION || !SESSION.round) return;
    if (CORR_HELD === null || ROUND_SENDING || AT) return;
    const tag = (event.target && event.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'BUTTON') return;
    if (event.key === 'Enter') { event.preventDefault(); roundSeal(); }
    else if (event.key === 'Escape') { event.preventDefault(); roundPutBack(); }
  });
}

// ----- on the board

// The standoff links over the board, from the round's state (`roundMarks`).
function standoffHtml(round, cell) {
  return roundMarks(round).map(mark => {
    const a = squareXY(mark.squares[0], cell, FRAME), b = squareXY(mark.squares[1], cell, FRAME);
    return standoffLine(a[0] + cell / 2, a[1] + cell / 2, b[0] + cell / 2, b[1] + cell / 2,
                        cell, mark.kind);
  }).join('');
}

// The legend's kinds for the standoff links drawn now.
function standoffKinds(round) {
  return [...new Set(roundMarks(round).map(mark => mark.kind))];
}

// The holder of the initiative (§3), beside their name, where the tie rule is the
// initiative: the core's state, drawn.
function initiativeHtml(player) {
  const round = SESSION && SESSION.round;
  if (!round || round.tie !== 'initiative' || round.initiative !== player) return '';
  return ` <span class="pill" data-initiative="${player}" title="decides the next tied
    collision in this player's favour, and then passes to the other player">initiative</span>`;
}

// ----- the reveal, animated

// How long a void move's step toward its target and back takes, and how far it reaches,
// are `PACE.voidStep` and `PACE.voidReach`.
//
// The round's first beat: both moves at once, drawn from `roundStory`. A collision's two
// pieces are timed to arrive together; a loser travels as a ghost and is struck where they
// met. Returns what `animateChange` reads from a beat: when it is busy until, when the last
// piece came to rest, and no ghosts for the wild pieces' phase.
function animateRound(svg, before, after, reveal, cell, W, H, view) {
  const story = roundStory(before, after, reveal);
  const at = sq => squareXY(sq, cell, view);
  let busy = 0, landed = 0;
  const blows = new Map();
  // A void piece taken where it stood steps out and back before the piece that takes it
  // arrives, so every travel waits for it.
  const lag = story.moves.some(m => m.kind === 'void' && m.taken) ? PACE.voidStep * 0.6 : 0;
  // Each travelling move's step, arc and flight.
  const legs = story.moves.filter(m => m.from !== null && (m.kind === 'arrive'
                                        || m.kind === 'win' || m.kind === 'lose')).map(m => {
    const allow = wrapAllow(SESSION.ruleset.wrap, m.player);
    const step = wrapStep(m.from, m.to, W, H, allow);
    const arc = pathHops(before, m.from, step, W, H, view);
    const [dx, dy] = stepXY(step, cell, view);
    const distance = Math.hypot(dx, dy);
    const flight = Math.min(PACE.travelMax, PACE.travelMin + distance * PACE.travelPerUnit);
    return Object.assign({ step, arc, dx, dy, flight }, m);
  });
  // Two pieces meeting arrive together: the shorter trip waits.
  const meet = Math.max(0, ...legs.filter(l => l.kind !== 'arrive').map(l => l.flight));
  for (const leg of legs) {
    const delay = lag + (leg.kind === 'arrive' ? 0 : meet - leg.flight);
    const [tx, ty] = at(leg.to);
    let node = null;
    if (leg.kind === 'lose') {
      const piece = pieceOf(before[leg.from]);
      if (!piece) continue;
      node = ghostAt(svg, piece, tx, ty, cell);
    } else {
      node = svg.querySelector(`[data-piece="${leg.to}"]`);
      if (!node) continue;
      svg.appendChild(node);
    }
    const flight = travel(node, -leg.dx, -leg.dy, leg.arc, delay);
    if (leg.step.wrapped) {
      animateCrossing(svg, node, pieceOf(before[leg.from]), ...at(leg.from), leg.dx, leg.dy,
                      cell, leg.arc, flight, delay);
    }
    const arrive = delay + flight;
    landed = Math.max(landed, arrive);
    busy = Math.max(busy, arrive);
    if (leg.kind === 'lose') {
      busy = Math.max(busy, strike(svg, node, tx, ty, cell,
                                   { at: arrive, dx: leg.dx, dy: leg.dy }));
    } else {
      blows.set(leg.to, { at: delay + flight * PACE.impact, dx: leg.dx, dy: leg.dy });
    }
  }
  // A void move: toward its target and back. The piece itself where it still stands; a
  // ghost of it where the other move took it there, struck when that piece arrives.
  for (const move of story.moves.filter(m => m.kind === 'void')) {
    const piece = pieceOf(before[move.from]);
    if (!piece) continue;
    const [fx, fy] = at(move.from);
    let node = move.taken ? ghostAt(svg, piece, fx, fy, cell)
      : svg.querySelector(`[data-piece="${move.from}"]`);
    if (!node) continue;
    svg.appendChild(node);
    const step = wrapStep(move.from, move.to, W, H, wrapAllow(SESSION.ruleset.wrap, move.player));
    const [dx, dy] = stepXY(step, cell, view);
    const reach = PACE.voidReach;
    node.animate([
      { transform: 'translate(0px, 0px)' },
      { transform: `translate(${dx * reach}px, ${dy * reach}px)`, offset: 0.45 },
      { transform: 'translate(0px, 0px)' },
    ], { duration: PACE.voidStep, easing: 'ease-in-out' });
    busy = Math.max(busy, PACE.voidStep);
    landed = Math.max(landed, PACE.voidStep);
    if (move.taken) {
      const blow = blows.get(move.from) || { at: Math.max(PACE.voidStep, landed), dx: 0, dy: 0 };
      busy = Math.max(busy, strike(svg, node, fx, fy, cell, blow));
      blows.delete(move.from);
    }
  }
  // A piece put down from the hand fades in, as in any game; one that lost a meeting was
  // drawn above.
  for (const move of story.moves.filter(m => m.from === null
                                           && (m.kind === 'drop' || m.kind === 'win'))) {
    const node = svg.querySelector(`[data-piece="${move.to}"]`);
    if (!node) continue;
    node.animate([{ opacity: 0 }, { opacity: 1 }],
                 { duration: PACE.drop, delay: lag, easing: 'ease-out' });
    landed = Math.max(landed, lag + PACE.drop);
    busy = Math.max(busy, lag + PACE.drop);
  }
  // Every other piece taken off: struck when the move that took it arrives, or, with no
  // move arriving there (a blast, a piece reborn elsewhere), when the last one lands.
  const ex = SESSION.ruleset.explosion;
  for (const square of story.taken) {
    const piece = pieceOf(before[square]);
    if (!piece) continue;
    const [x, y] = at(square);
    const blow = blows.get(square) || { at: landed, dx: 0, dy: 0 };
    busy = Math.max(busy, ex && !blows.has(square)
      ? explode(svg, piece, x, y, cell, blow) : capture(svg, piece, x, y, cell, blow));
  }
  return { busy: busy, landed: landed, doomed: new Map() };
}
