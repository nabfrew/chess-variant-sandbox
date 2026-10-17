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
// its target and comes back; a swap that met (schema 27) draws the two meeting half-way,
// the loser, or both, broken there, and a clash between the two squares after. The round's
// sentences are the server's (`round_reveal`), in
// the slot under the board and in the move list.
//
// **The standoff.** Two pieces that could only take each other by a capture-only move, after
// one void round (the core's `voidRun`) or two in a row (`bars`), are linked on the board
// (`roundMarks`, `standoffLine`), so a person knows why the capture is not offered.
//
// **The rest** (schema 27, `rest`). The piece each player moved last round wears a badge
// (`roundRests`, `restBadge`): it may not move this round, so the board offers none of its
// moves, unless it is all that player has.
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

// Schema 27 (Q942b, `swap: "collision"`): the swap that met in a revealed round, from the
// core's event (`lastMove.reveal`), or null: its two squares, player 0's origin first, and
// the winner (-1 where both were lost). The event's square is player 0's destination; the
// other square is the one player 0 left, which player 1 went for.
function roundClash(last) {
  const reveal = last && last.reveal;
  if (!reveal) return null;
  const swap = (reveal.events || []).find(event => event.kind === 'swap');
  const first = (reveal.moves || []).find(move => move.player === 0);
  if (!swap || !first || first.from === null || first.from === undefined) return null;
  return { squares: [first.from, first.to], winner: swap.player };
}

// Schema 27 (Q942a, `rest`): the rested squares the round now open carries, player 0's
// first, the core's state: `[{player, square}]`, empty under any game without the rule.
function roundRests(round) {
  if (!round || !round.rest) return [];
  return round.rest.map((square, player) => ({ player, square }))
    .filter(rest => rest.square !== null && rest.square !== undefined);
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
//
// A swap that met (schema 27, `swap: "collision"`) is its own meeting, on no one square:
// both moves are `win` or `lose` with `clash: true`, and `clash` is `roundClash`'s.
function roundStory(before, after, reveal) {
  const events = (reveal && reveal.events) || [];
  const collision = events.find(event => event.kind === 'collision') || null;
  const meeting = collision ? { square: collision.square, winner: collision.player } : null;
  const clash = roundClash({ reveal });
  const moves = ((reveal && reveal.moves) || []).map(move => {
    const out = { player: move.player, from: move.from, to: move.to };
    if (clash && move.from !== null && move.from !== undefined && !move.void) {
      out.kind = clash.winner === move.player ? 'win' : 'lose';
      out.clash = true;
    } else if (move.from === null || move.from === undefined) {
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
  return { moves: moves, taken: taken, meeting: meeting, clash: clash };
}

// One row of the move list for a round: its number, both moves in seat order with what
// each came to, and the round's sentences under them.
function roundRowHtml(number, first, second) {
  // Schema 27: the move a spy game's announced round announced, marked as such.
  const announcer = second && second.reveal ? second.reveal.announcer : undefined;
  const shown = move => move && announcer !== undefined && announcer !== null
    && move.player === announcer
    ? ' <span class="muted small" data-announced title="announced first, and shown to the other player before they answered">announced</span>'
    : '';
  const said = move => {
    const result = move && move.result;
    if (!result) return '';
    const mark = result === 'void' ? 'void' : result === 'won' ? 'won'
      : result === 'lost' ? '✕ lost' : '✕ both lost';
    return ` <span class="muted small" data-result="${esc(result)}">${esc(mark)}</span>`;
  };
  const words = ((second && second.reveal && second.reveal.words) || [])
    .map(line => `<div class="muted small">${esc(line)}</div>`).join('');
  return `${esc(number)}. ${esc(first.text)}${shown(first)}${said(first)}${lookedHtml(first)} <span
    class="muted">|</span> ${esc(second.text)}${shown(second)}${said(second)}${
    lookedHtml(second)}${words}`;
}

// ----- the strip and the slot

// The strip's words while a round is open. `SESSION.round.choosing` is the person the board
// offers moves to; the other seat is the computer, or the other person at this screen.
function roundStatusHtml() {
  const round = SESSION.round;
  const number = `<span class="muted">Round ${round.number}.</span> `;
  const held = corrHeldEntry();
  // Schema 27: a spy game's announced round has an order, and its own words.
  if (round.announcer !== undefined && round.announcer !== null && !SESSION.correspondence) {
    return announcedStatusHtml(number, held);
  }
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
  // A spy game's blind rounds say why they are blind (schema 27).
  const spy = SESSION.spy ? `<span class="frame-note" data-spy="${esc(SESSION.spy.phase)}">${
    esc(spyLine(SESSION.spy, round, seat, SESSION.ruleset.sides))}</span>` : '';
  return `${number}${seatDot(seat)}${who(seat)} to choose. ${theirs}${note}${spy}`;
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
  const round = SESSION.round;
  // Schema 27: a spy game's announced round, announced first and then answered.
  const announcer = between ? null : round.announcer;
  const announced = announcer !== undefined && announcer !== null;
  let says, row = '';
  if (ROUND_SENDING) {
    says = '<b>Your move is in.</b> Waiting for the reveal.';
  } else if (announced) {
    [says, row] = announcedSendHtml(round, held);
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

// The slot of an announced round (schema 27): what to do, the announced move once it is in,
// and with a move held, *Announce* or *Answer* and *Change my move*. `[says, row]`.
function announcedSendHtml(round, held) {
  const announcer = round.announcer, holder = 1 - announcer;
  const shown = round.announced;
  const theirs = esc(sideName(holder));
  if (held) {
    const button = shown ? 'Answer' : 'Announce';
    const says = shown
      ? `<b>Your answer is not made yet.</b> To change it, tap another square or press <b>Change
        my move</b>.<span class="more"> Both moves are then made at once.</span>`
      : `<b>Not announced yet.</b> To change it, tap another square or press <b>Change my
        move</b>.<span class="more"> Once it is announced, ${theirs} sees it before choosing.</span>`;
    const row = `<button id="sealHeld" class="primary send-big" title="${shown
      ? 'answer with it' : 'announce it'} (Enter)">${button}</button><button id="changeHeld"
      class="send-change" title="put it back (Esc)">Change my move</button>`;
    return [says, row];
  }
  if (round.choosing === announcer) {
    return [`<b>Announce your move:</b> pick it on the board, then press <b>Announce</b>.<span
      class="more"> ${theirs} sees it before choosing theirs; then both are made at once.</span>`, ''];
  }
  if (round.choosing === holder && shown) {
    return [`<b>${esc(sideName(announcer))} announced ${esc(shown.text)}.</b> Pick your answer
      on the board, then press <b>Answer</b>.<span class="more"> Both moves are made at
      once.</span>`, ''];
  }
  return [esc(spyLine(SESSION.spy, round, SESSION.sideToMove, SESSION.ruleset.sides)), ''];
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

// A swap that met (`roundClash`), drawn between its two squares (`clashMark`).
function clashHtml(clash, cell) {
  if (!clash) return '';
  const a = squareXY(clash.squares[0], cell, FRAME), b = squareXY(clash.squares[1], cell, FRAME);
  return clashMark(a[0] + cell / 2, a[1] + cell / 2, b[0] + cell / 2, b[1] + cell / 2, cell);
}

// Each rested piece's badge (`restBadge`), on its square's corner.
function restsHtml(rests, cell) {
  return rests.map(rest => {
    const at = squareXY(rest.square, cell, FRAME);
    return restBadge(at[0], at[1], cell, rest.player);
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
  const meet = Math.max(0, ...legs.filter(l => l.kind !== 'arrive' && !l.clash)
    .map(l => l.flight));
  // Two pieces of a swap that met (schema 27) pass the middle of their two squares together:
  // a loser travels half its way, as a ghost, and is struck there; a winner goes on through.
  const half = l => Math.min(PACE.travelMax,
                             PACE.travelMin + Math.hypot(l.dx, l.dy) / 2 * PACE.travelPerUnit);
  const midway = l => (l.kind === 'lose' ? half(l) : l.flight / 2);
  const clashAt = Math.max(0, ...legs.filter(l => l.clash).map(midway));
  for (const leg of legs) {
    const delay = lag + (leg.clash ? clashAt - midway(leg)
      : leg.kind === 'arrive' ? 0 : meet - leg.flight);
    let [tx, ty] = at(leg.to);
    let dx = leg.dx, dy = leg.dy;
    let node = null;
    if (leg.kind === 'lose') {
      const piece = pieceOf(before[leg.from]);
      if (!piece) continue;
      if (leg.clash) {
        const [fx, fy] = at(leg.from);
        tx = (fx + tx) / 2; ty = (fy + ty) / 2; dx /= 2; dy /= 2;
      }
      node = ghostAt(svg, piece, tx, ty, cell);
    } else {
      node = svg.querySelector(`[data-piece="${leg.to}"]`);
      if (!node) continue;
      svg.appendChild(node);
    }
    const flight = travel(node, -dx, -dy, leg.arc, delay);
    if (leg.step.wrapped && !(leg.clash && leg.kind === 'lose')) {
      animateCrossing(svg, node, pieceOf(before[leg.from]), ...at(leg.from), leg.dx, leg.dy,
                      cell, leg.arc, flight, delay);
    }
    const arrive = delay + flight;
    landed = Math.max(landed, arrive);
    busy = Math.max(busy, arrive);
    if (leg.kind === 'lose') {
      busy = Math.max(busy, strike(svg, node, tx, ty, cell, { at: arrive, dx: dx, dy: dy }));
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

// ----- the spy (schema 27, Q935c, `docs/plan-spy.md` §3.6)
//
// A spy game is played by turns while both spies stand, then in rounds: **announced** ones
// while one player alone has a spy (the other, the announcer, chooses first and in the open,
// and the holder answers knowing the move), **blind** ones once both are lost. The phase is
// the core's (`SESSION.spy`, `Engine.phase`); the page words it and draws the announced move,
// and works nothing out. A turn is the alternating board as it is; a round is this file's.

// The phase line: what the phase is, in one sentence, from the core's phase (`spy`), the
// round's announced move (`round.announced`), the side to move (in a reply turn, the player
// whose spy was just lost) and the two sides' names. Plain text, for the strip and the slot.
function spyLine(spy, round, mover, names) {
  if (!spy) return '';
  const name = player => (names || [])[player] || `player ${player}`;
  if (spy.phase === 'alternating') return 'Both spies stand: the players take turns.';
  if (spy.phase === 'reply') {
    const lost = name(mover);
    return `${lost}'s spy is lost. ${lost} replies with one ordinary move; then rounds begin, `
      + `and ${lost} announces first.`;
  }
  if (spy.phase === 'announced') {
    const announcer = name(spy.announcer), holder = name(1 - spy.announcer);
    const shown = round && round.announced;
    if (shown) return `${announcer} announced ${shown.text}; ${holder} answers.`;
    return `Only ${holder} has a spy: ${announcer} announces a move first, and ${holder} `
      + 'answers knowing it.';
  }
  if (spy.phase === 'blind') return 'Both spies are lost: both players choose in secret.';
  return '';
}

// The announced move as the board draws it, from the round now open (`round.announced`, the
// server's description of the move, and `round.announcer`): `{from, to, dest, player, text}`
// and a transit's two ends, or null where nothing is announced.
function announcedArrow(round) {
  const shown = round && round.announced;
  if (!shown) return null;
  const from = shown.from === null || shown.from === undefined ? null : shown.from;
  const dest = shown.landing !== undefined && shown.landing !== null ? shown.landing : shown.to;
  const arrow = { from: from, to: shown.to, dest: dest, player: round.announcer, text: shown.text };
  if (shown.portal !== undefined && shown.portal !== null) {
    arrow.portal = shown.portal;
    arrow.portalOut = shown.portalOut;
  }
  return arrow;
}

// The announced move on the board (`announcedArrow`): a solid ring on the square it goes to
// and a solid arrow there along the mover's own step (`announceLine`), in two legs through
// a portal; a drop is its ring alone. Under the held move, which is drawn over everything.
function announcedHtml(arrow, cell, W, H, wrap) {
  if (!arrow) return '';
  const f = n => n.toFixed(3);
  const at = squareXY(arrow.dest, cell, FRAME);
  let out = `<g data-mark="announced" pointer-events="none">${cellShape(at[0], at[1], cell, FRAME,
    `fill="none" stroke="var(--held-halo)" stroke-width="${f(cell * 0.12)}" opacity="0.9"`, 0.06)}${
    cellShape(at[0], at[1], cell, FRAME, `fill="none" stroke="var(--announced)"
      stroke-width="${f(cell * 0.065)}"`, 0.06)}</g>`;
  if (arrow.from === null || arrow.from === arrow.dest) return out;
  const allow = wrapAllow(wrap, arrow.player);
  const legs = arrow.portal !== undefined && arrow.portalOut !== undefined
    ? [[arrow.from, arrow.portal], [arrow.portalOut, arrow.dest]] : [[arrow.from, arrow.dest]];
  for (const [from, to] of legs) {
    const start = squareXY(from, cell, FRAME);
    const step = stepXY(wrapStep(from, to, W, H, allow), cell, FRAME);
    const length = Math.hypot(step[0], step[1]);
    if (!(length > cell * 0.6)) continue;
    const ux = step[0] / length, uy = step[1] / length;
    const cx = start[0] + cell / 2, cy = start[1] + cell / 2;
    out += announceLine(cx + ux * cell * 0.3, cy + uy * cell * 0.3,
                        cx + ux * (length - cell * 0.3), cy + uy * (length - cell * 0.3), cell);
  }
  return out;
}

// The phase line in the strip of a turn, as a note after whose move it is; nothing in a
// round, whose own strip says the phase (`roundStatusHtml`), or in any other game.
function spyNoteHtml() {
  const spy = SESSION && SESSION.spy;
  if (!spy || SESSION.round) return '';
  return `<span class="frame-note" data-spy="${esc(spy.phase)}">${
    esc(spyLine(spy, null, SESSION.sideToMove, SESSION.ruleset.sides))}</span>`;
}

// The slot under the board of a spy game while it is played by turns: the phase, and what
// comes next. It is there from the first move, so the board does not move when the rounds
// begin and the slot takes their buttons.
function drawSpySend(slot) {
  if (SESSION.round) { drawRoundSend(slot); return; }
  if (SESSION.outcome) {
    slot.hidden = true;
    slot.innerHTML = '';
    return;
  }
  slot.hidden = false;
  slot.classList.toggle('held', false);
  const mover = SESSION.sideToMove;
  const lost = esc(sideName(mover)), holder = esc(sideName(1 - mover));
  const says = SESSION.spy.phase === 'reply'
    ? `<b>${lost}'s spy is lost.</b> ${lost} replies with one ordinary move; then rounds
      begin.<span class="more"> Each round ${lost}, without a spy, announces a move first, and
      ${holder} answers knowing it.</span>`
    : `<b>Both spies stand:</b> the players take turns.<span class="more"> Once a spy is lost,
      play turns to rounds: the player without a spy announces each move first, and the other
      answers knowing it.</span>`;
  slot.innerHTML = `<div class="send-says" data-spy="${esc(SESSION.spy.phase)}">${says}</div>`;
}

// The strip of an announced round: whose choice it is, and once the announcement is in,
// what it was. Two people at one screen pass it on after the announcement.
function announcedStatusHtml(number, held) {
  const round = SESSION.round;
  const announcer = round.announcer, holder = 1 - announcer;
  const shown = round.announced;
  if (ROUND_SENDING) {
    return shown ? `${number}<b>Your answer is in:</b> waiting for the reveal.`
      : `${number}<b>Your announcement is in.</b>`;
  }
  if (held) {
    return shown
      ? `${number}<b>${esc(held.text)}</b> is your answer to ${esc(shown.text)}, not made yet.
        You can change it until you answer.`
      : `${number}<b>${esc(held.text)}</b> is your choice, not announced yet. You can change it
        until you announce it.`;
  }
  const both = SESSION.seats[0] === 'human' && SESSION.seats[1] === 'human';
  const who = player => esc(sideName(player)) + (both ? '' : ` (${esc(seatLabel(player))})`);
  const note = NOTE ? `<span class="frame-note">${esc(NOTE)}</span>` : '';
  const seat = round.choosing;
  if (seat === announcer) {
    return `${number}${seatDot(announcer)}${who(announcer)} to announce a move.<span
      class="frame-note">${esc(sideName(holder))} holds the only spy and answers knowing
      it.</span>${note}`;
  }
  if (seat === holder && shown) {
    return `${number}${seatDot(holder)}${esc(sideName(announcer))} announced <b>${
      esc(shown.text)}</b>; ${who(holder)} answers.${both ? `<span class="frame-note">Pass the
      screen to ${esc(sideName(holder))}.</span>` : ''}${note}`;
  }
  return `${number}${esc(spyLine(SESSION.spy, round, SESSION.sideToMove, SESSION.ruleset.sides))}`;
}
