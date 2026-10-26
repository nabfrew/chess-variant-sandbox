// A game played in rounds, on the play board (schema 26, Q918c, `docs/archive/plan-simultaneous.md`
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
// **The rest** (schema 29, `pieces[].rest`; schema 27's round rest before it). A resting
// piece wears a badge (`positionRests`, `restBadge`) from the position's `resting`, in any
// game: it may not move this turn, so the board offers none of its moves, unless it is all
// that player has. Where the game's longest rest is over 1 the badge carries the turns left
// (`restNumbered`, Q983b); a step back badges its own position, from the move's `resting`.
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

// Schema 29 (Q983a, `pieces[].rest`): the resting pieces of the position on screen, the
// core's state (`resting`, `[{player, square, left}]`, player 0's first), in any game,
// alternating or in rounds; empty under any game without a rest.
function positionRests(resting) {
  return (resting || []).filter(rest => rest.square !== null && rest.square !== undefined);
}

// Schema 29 (Q983b, `docs/plan-rest.md` §5.6): whether the pause signs carry the turns left,
// which they do where some piece rests more than one turn (the compiled `ruleset.rest`, each
// type's); with every rest at 1 the sign alone says it.
function restNumbered(rests) {
  return (rests || []).some(turns => turns > 1);
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
  // Schema 28: a round between two speeds is two moves in order, its own story.
  const order = events.find(event => event.kind === 'order');
  if (order) return orderedStory(before, after, reveal, order.player);
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

// Schema 28 (Q943b, `docs/archive/plan-speed.md` §4.5): a round between pieces of different speeds,
// as motions in two beats. The core's events say who moved first (`order`) and what came of
// the slower move (`lost`, `unplayed`, `void`, or nothing: it was made); the two boards say
// what was taken. Each move is `roundStory`'s `{player, from, to, kind}` with its `phase`, 1
// for the faster and 2 for the slower, and `ordered` is the faster player.
//
// The faster move is `arrive`, `drop` or `turn`, or `fell`: it arrived, and the slower move
// took it there (§2.5 C), so it travels as a ghost and is struck when the slower arrives.
// The slower move is `arrive`, `drop`, `turn` or `void` as at equal speeds, or `lost` (its
// piece taken, moved or changed first) or `unplayed` (the faster move ended the game); a lost
// or unplayed move whose piece's square the faster move changed is `taken: true`, drawn as a
// fading arrow from its square to the square it was sealed for (the user, §4.5), and an
// unplayed one whose piece still stands is a struck-through ghost where it was going.
function orderedStory(before, after, reveal, first) {
  const events = (reveal && reveal.events) || [];
  const said = kind => new Set(events.filter(event => event.kind === kind)
    .map(event => event.player));
  const lost = said('lost'), unplayed = said('unplayed'), voids = said('void');
  const moves = ((reveal && reveal.moves) || []).map(move => {
    const from = move.from === undefined ? null : move.from;
    const out = { player: move.player, from: from, to: move.to,
                  phase: move.player === first ? 1 : 2 };
    if (out.phase === 2 && (lost.has(move.player) || unplayed.has(move.player))) {
      out.kind = lost.has(move.player) ? 'lost' : 'unplayed';
      out.taken = from !== null && before[from] !== after[from];
    } else if (out.phase === 2 && voids.has(move.player)) {
      out.kind = 'void';
      out.taken = false;
    } else if (from === null) {
      out.kind = 'drop';
    } else if (from === move.to) {
      out.kind = 'turn';
    } else {
      out.kind = 'arrive';
    }
    return out;
  });
  const fast = moves.find(m => m.phase === 1), slow = moves.find(m => m.phase === 2);
  if (fast && slow && slow.kind === 'arrive' && slow.to === fast.to
      && (fast.kind === 'arrive' || fast.kind === 'drop')) {
    fast.kind = 'fell';
  }
  // A square a piece left, and a void or turned piece's own, are not captures.
  const left = new Set(moves.filter(m => (m.kind === 'arrive' || m.kind === 'fell')
                                         && m.from !== null).map(m => m.from));
  const standing = new Set(moves.filter(m => m.kind === 'void' || m.kind === 'turn')
    .map(m => m.from));
  const taken = [];
  for (let square = 0; square < before.length; square++) {
    if (!before[square] || before[square] === after[square]) continue;
    if (left.has(square) || standing.has(square)) continue;
    taken.push(square);
  }
  return { moves: moves, taken: taken, meeting: null, clash: null, ordered: first };
}

// The order marks of the last round on screen (`orderBadge`): `[{square, n, player}]`, the
// faster move's on the square it ended on, the slower's where it ended or, void, where it
// stood; none for a slower move lost or never played, or in any round of equal speeds.
function roundOrder(last) {
  const reveal = last && last.reveal;
  const order = reveal && (reveal.events || []).find(event => event.kind === 'order');
  if (!order) return [];
  const gone = new Set((reveal.events || []).filter(event => event.kind === 'lost'
    || event.kind === 'unplayed').map(event => event.player));
  const marks = [];
  for (const move of reveal.moves || []) {
    if (gone.has(move.player)) continue;
    const square = move.void && move.from !== null && move.from !== undefined
      ? move.from : move.to;
    marks.push({ square: square, n: move.player === order.player ? 1 : 2, player: move.player });
  }
  return marks.sort((a, b) => a.n - b.n);
}

// The speeds of a game, as the legend's line says them: each rank from the fastest, the
// pieces at it by name, `[{speed, names}]`; empty in a game without a speed. `speed` is the
// compiled `[type]` table (`ruleset.speed`), `names` the pieces' names; a type without a
// name is no piece a player moves.
function speedRanks(speed, names) {
  if (!speed || !speed.length) return [];
  const ranks = new Map();
  speed.forEach((value, type) => {
    const name = (names || [])[type];
    if (!name) return;
    if (!ranks.has(value)) ranks.set(value, []);
    ranks.get(value).push(name);
  });
  return [...ranks.entries()].sort((a, b) => b[0] - a[0])
    .map(([value, at]) => ({ speed: value, names: at }));
}

// A piece's speed as its card's hover line says it (§4.5): *speed 2: moves before speeds 0
// and 1*. `ranks` are the speeds the game uses.
function speedLine(speed, ranks) {
  const list = items => (items.length < 2 ? `${items[0]}`
    : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`);
  const word = items => (items.length > 1 ? 'speeds' : 'speed');
  const slower = [...new Set(ranks)].filter(rank => rank < speed).sort((a, b) => a - b);
  const faster = [...new Set(ranks)].filter(rank => rank > speed).sort((a, b) => a - b);
  const before = slower.length ? `moves before ${word(slower)} ${list(slower)}` : '';
  const after = faster.length ? `after ${word(faster)} ${list(faster)}` : '';
  if (before && after) return `speed ${speed}: ${before}, ${after}`;
  if (before) return `speed ${speed}: ${before}`;
  if (after) return `speed ${speed}: moves ${after}`;
  return `speed ${speed}`;
}

// The legend's line of speeds (§4.5): *speeds: knight 2; bishop, king 1; pawn 0*, and what a
// speed does, or '' in a game without one.
function speedLegendHtml(speed, names) {
  const ranks = speedRanks(speed, names);
  if (!ranks.length) return '';
  const said = ranks.map(rank => `${rank.names.map(esc).join(', ')} ${rank.speed}`).join('; ');
  return `<span class="entry" data-legend="speed"><b>speeds</b> ${said}: in a round the
    faster piece's move is made first, and pieces of one speed move at once</span>`;
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
      : result === 'lost' ? '✕ lost' : result === 'unplayed' ? 'not played' : '✕ both lost';
    return ` <span class="muted small" data-result="${esc(result)}">${esc(mark)}</span>`;
  };
  // Schema 28: which move was made first, in a round between two speeds (`12. Nc3¹ | Qxc3²`).
  const order = second && second.reveal
    && (second.reveal.events || []).find(event => event.kind === 'order');
  const nth = move => {
    if (!order || !move) return '';
    const n = move.player === order.player ? 1 : 2;
    return `<sup data-order="${n}" title="${n === 1 ? 'made first: the faster piece'
      : 'made second: the slower piece'}">${n}</sup>`;
  };
  const words = ((second && second.reveal && second.reveal.words) || [])
    .map(line => `<div class="muted small">${esc(line)}</div>`).join('');
  return `${esc(number)}. ${esc(first.text)}${nth(first)}${shown(first)}${said(first)}${
    lookedHtml(first)} <span class="muted">|</span> ${esc(second.text)}${nth(second)}${
    shown(second)}${said(second)}${
    lookedHtml(second)}${words}`;
}

// ----- the strip and the slot

// The strip's words while a round is open. `SESSION.round.choosing` is the person the board
// offers moves to; the other seat is the computer, or the other person at this screen.
function roundStatusHtml() {
  const round = SESSION.round;
  const number = `<span class="muted">Round ${round.number}.</span> `;
  const held = corrHeldEntry();
  // Schema 27: a spy game's announced round has an order, and its own words; between two
  // people at two screens (Q935e), said to the reader by name.
  if (round.announcer !== undefined && round.announcer !== null) {
    return SESSION.correspondence ? announcedCorrStatusHtml(number, held)
      : announcedStatusHtml(number, held);
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
  // Schema 27: a spy game's announced round, announced first and then answered; by
  // correspondence too (Q935e), where the reader is one of the two.
  const announcer = round.announcer;
  const announced = announcer !== undefined && announcer !== null;
  let says, row = '';
  if (ROUND_SENDING) {
    says = '<b>Your move is in.</b> Waiting for the reveal.';
  } else if (announced) {
    [says, row] = between ? announcedCorrSendHtml(round, held, between)
      : announcedSendHtml(round, held);
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

// The slot of an announced round between two people at two screens (Q935e): the reader's
// part in it by name -- announce, wait for the answer, wait for the announcement, answer
// it -- and with a move held, *Announce* or *Answer* and *Change my move*. `[says, row]`.
function announcedCorrSendHtml(round, held, between) {
  const you = between.you, announcer = round.announcer, shown = round.announced;
  const other = esc(between.players[1 - you]);
  if (held) {
    const button = shown ? 'Answer' : 'Announce';
    const says = shown
      ? `<b>Your answer is not sent yet.</b> To change it, tap another square or press <b>Change
        my move</b>.<span class="more"> Both moves are then made at once.</span>`
      : `<b>Not announced yet.</b> To change it, tap another square or press <b>Change my
        move</b>.<span class="more"> Once it is announced, ${other} sees it before choosing.</span>`;
    const row = `<button id="sealHeld" class="primary send-big" title="${shown
      ? 'answer with it' : 'announce it'} (Enter)">${button}</button><button id="changeHeld"
      class="send-change" title="put it back (Esc)">Change my move</button>`;
    return [says, row];
  }
  if (you === announcer) {
    return shown
      ? [`<b>Your announcement is made:</b> ${esc(shown.text)}. ${other} answers it.<span
        class="more"> You can close this page.</span>`, '']
      : [`<b>Announce your move:</b> pick it on the board, then press <b>Announce</b>.<span
        class="more"> ${other} sees it before choosing theirs; then both are made at once.</span>`, ''];
  }
  return shown
    ? [`<b>${other} announced ${esc(shown.text)}.</b> Pick your answer on the board, then press
      <b>Answer</b>.<span class="more"> Both moves are made at once.</span>`, '']
    : [`<b>${other} announces first.</b> You answer once their move is in.<span class="more">
      You can close this page; the game waits for you.</span>`, ''];
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

// Each rested piece's badge (`restBadge`), on its square's corner, with its turns left where
// `numbered` (`restNumbered`).
function restsHtml(rests, cell, numbered) {
  return rests.map(rest => {
    const at = squareXY(rest.square, cell, FRAME);
    return restBadge(at[0], at[1], cell, rest.player, numbered ? rest.left : undefined);
  }).join('');
}

// A lost move's arrow (schema 28, §4.5), from the middle of its piece's square to the middle
// of the square it was sealed for, along the mover's own step: the held move's dashed pine
// cased in `--held-halo`, bolder than a held arrow and reaching the squares' middles, so a
// one-square move still reads while the faster piece lands on it.
function lostArrowHtml(from, to, cell, W, H, allow, view) {
  const f = n => n.toFixed(3);
  const [x, y] = squareXY(from, cell, view);
  const step = stepXY(wrapStep(from, to, W, H, allow), cell, view);
  const length = Math.hypot(step[0], step[1]);
  if (!(length > 0)) return '';
  const ux = step[0] / length, uy = step[1] / length;
  const x1 = x + cell / 2 + ux * cell * 0.08, y1 = y + cell / 2 + uy * cell * 0.08;
  const x2 = x + cell / 2 + ux * (length - cell * 0.12), y2 = y + cell / 2 + uy * (length - cell * 0.12);
  const head = cell * 0.3;
  const back = [x2 - ux * head, y2 - uy * head];
  const wing = [-uy * head * 0.7, ux * head * 0.7];
  const tip = `M${f(back[0] + wing[0])} ${f(back[1] + wing[1])}L${f(x2)} ${f(y2)}L${
    f(back[0] - wing[0])} ${f(back[1] - wing[1])}Z`;
  const line = `x1="${f(x1)}" y1="${f(y1)}" x2="${f(back[0])}" y2="${f(back[1])}"`;
  return `<line ${line} stroke="var(--held-halo)" stroke-width="${f(cell * 0.19)}"/><path
      d="${tip}" fill="var(--held-halo)" stroke="var(--held-halo)" stroke-width="${f(cell * 0.1)}"
      stroke-linejoin="round"/><line ${line} stroke="var(--held)" stroke-width="${f(cell * 0.1)}"
      stroke-dasharray="${f(cell * 0.16)} ${f(cell * 0.09)}"/><path d="${tip}" fill="var(--held)"
      stroke="var(--held)" stroke-width="${f(cell * 0.03)}" stroke-linejoin="round"/>`;
}

// Each move's order mark in the last round on screen (`roundOrder`, `orderBadge`), on its
// square's bottom corner; the second on the other corner where both moves ended on one
// square (§2.5 C: the slower took the faster where it had landed).
function ordersHtml(marks, cell) {
  return marks.map((mark, i) => {
    const at = squareXY(mark.square, cell, FRAME);
    const shared = i > 0 && marks[i - 1].square === mark.square;
    return orderBadge(at[0], at[1], cell, mark.n, mark.player, shared);
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
  if (story.ordered !== undefined) return animateOrdered(svg, before, after, story, cell, W, H, view);
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
      ? Math.max(blastFire(svg, [square], cell, view, blow.at),
                 burn(svg, piece, x, y, cell, blow, null))
      : capture(svg, piece, x, y, cell, blow));
  }
  return { busy: busy, landed: landed, doomed: new Map() };
}

// Schema 28 (Q943b, `docs/archive/plan-speed.md` §4.5; the user: *"they should move in the sequence
// the mechanics dictate"*): an ordered round in two beats, from `orderedStory`. The faster
// move completes first, with what it takes; then, `PACE.beat` after it lands, the slower:
// made, or its void step-and-return. A slower move lost with its piece is an arrow from that
// piece's square to the square it was sealed for, fading out as the faster capture lands on
// it; one never played while its piece stands is that piece's ghost where it was going,
// struck through. Each moving piece's order mark (`orderBadge`, drawn with the board) fades
// in as its move lands. Returns what `animateChange` reads, as `animateRound` does.
function animateOrdered(svg, before, after, story, cell, W, H, view) {
  const at = sq => squareXY(sq, cell, view);
  const centre = sq => { const [x, y] = at(sq); return [x + cell / 2, y + cell / 2]; };
  const fast = story.moves.find(m => m.phase === 1);
  const slow = story.moves.find(m => m.phase === 2);
  const blows = new Map();
  let busy = 0;
  const badge = (move, when) => {
    const node = svg.querySelector(`[data-mark="order"][data-player="${move.player}"]`);
    if (node && typeof node.animate === 'function') {
      node.animate([{ opacity: 0 }, { opacity: 1 }],
                   { duration: PACE.beat, delay: when, fill: 'backwards' });
    }
  };
  // One move's travel from its square to where it ended, starting at `delay`: the piece the
  // board drew there, or a ghost of it where the other move then took it (`fell`). Returns
  // when it lands and when it strikes what stood there, or null for no travel.
  const fly = (move, delay) => {
    if (move.from === null || move.from === move.to) return null;
    const allow = wrapAllow(SESSION.ruleset.wrap, move.player);
    const step = wrapStep(move.from, move.to, W, H, allow);
    const arc = pathHops(before, move.from, step, W, H, view);
    const [dx, dy] = stepXY(step, cell, view);
    let node;
    if (move.kind === 'fell') {
      const piece = pieceOf(before[move.from]);
      if (!piece) return null;
      node = ghostAt(svg, piece, ...at(move.to), cell);
    } else {
      node = svg.querySelector(`[data-piece="${move.to}"]`);
      if (!node) return null;
      svg.appendChild(node);
    }
    const flight = travel(node, -dx, -dy, arc, delay);
    if (step.wrapped) {
      animateCrossing(svg, node, pieceOf(before[move.from]), ...at(move.from), dx, dy, cell, arc,
                      flight, delay);
    }
    if (!blows.has(move.to)) blows.set(move.to, { at: delay + flight * PACE.impact, dx, dy });
    return { node, landed: delay + flight, impact: delay + flight * PACE.impact, dx, dy };
  };
  const fade = (move, delay) => {
    const node = svg.querySelector(`[data-piece="${move.to}"]`);
    if (!node) return delay;
    node.animate([{ opacity: 0 }, { opacity: 1 }],
                 { duration: PACE.drop, delay: delay, fill: 'backwards', easing: 'ease-out' });
    return delay + PACE.drop;
  };

  // Beat one: the faster move alone.
  let first = 0, fell = null;
  if (fast) {
    if (fast.kind === 'drop') first = fade(fast, 0);
    else {
      const went = fly(fast, 0);
      if (went) first = went.landed;
      if (went && fast.kind === 'fell') fell = went;
    }
    badge(fast, first);
  }
  busy = Math.max(busy, first);
  const second = first + PACE.beat;

  // Beat two: the slower move, or what became of it.
  let last = first, over = null;
  if (slow) {
    const arrow = (slow.kind === 'lost' || slow.kind === 'unplayed') && slow.taken;
    if (arrow && slow.from !== null && slow.from !== slow.to) {
      // The sealed move it never made, fading as the faster capture lands on its piece.
      const blow = blows.get(slow.from) || { at: first };
      const allow = wrapAllow(SESSION.ruleset.wrap, slow.player);
      const group = document.createElementNS(SVGNS, 'g');
      group.setAttribute('data-reveal', 'lost');
      group.setAttribute('pointer-events', 'none');
      group.setAttribute('stroke-linecap', 'round');
      group.innerHTML = lostArrowHtml(slow.from, slow.to, cell, W, H, allow, view);
      svg.appendChild(group);
      const span = blow.at + PACE.strike;
      const gone = group.animate([
        { opacity: 0.95 }, { opacity: 0.95, offset: Math.min(0.9, blow.at / span) },
        { opacity: 0 },
      ], { duration: span, easing: 'ease-in', fill: 'forwards' });
      gone.onfinish = () => group.remove();
      busy = Math.max(busy, span);
      over = group;
    } else if (slow.kind === 'unplayed') {
      // Never played, its piece still standing: where it was going, struck through.
      const piece = pieceOf(before[slow.from === null ? slow.to : slow.from]);
      if (piece) {
        const [x, y] = at(slow.to);
        const ghost = document.createElementNS(SVGNS, 'g');
        ghost.setAttribute('data-reveal', 'unplayed');
        ghost.setAttribute('pointer-events', 'none');
        ghost.innerHTML = `<g opacity="0.55">${drawPiece(piece, x, y, cell)}</g><line x1="${
          x + cell * 0.18}" y1="${y + cell * 0.82}" x2="${x + cell * 0.82}" y2="${y + cell * 0.18}"
          stroke="var(--bad)" stroke-width="${cell * 0.07}" stroke-linecap="round"/>`;
        svg.appendChild(ghost);
        const hold = 1400;
        const shown = ghost.animate([
          { opacity: 0 }, { opacity: 1, offset: 0.2 }, { opacity: 1, offset: 0.8 }, { opacity: 0 },
        ], { duration: hold, delay: second, fill: 'both' });
        shown.onfinish = () => ghost.remove();
        busy = Math.max(busy, second + hold);
      }
    } else if (slow.kind === 'void') {
      const node = slow.from === null ? null : svg.querySelector(`[data-piece="${slow.from}"]`);
      if (node) {
        svg.appendChild(node);
        const step = wrapStep(slow.from, slow.to, W, H, wrapAllow(SESSION.ruleset.wrap, slow.player));
        const [dx, dy] = stepXY(step, cell, view);
        const reach = PACE.voidReach;
        node.animate([
          { transform: 'translate(0px, 0px)' },
          { transform: `translate(${dx * reach}px, ${dy * reach}px)`, offset: 0.45 },
          { transform: 'translate(0px, 0px)' },
        ], { duration: PACE.voidStep, delay: second, easing: 'ease-in-out' });
        last = second + PACE.voidStep;
      }
      badge(slow, second);
    } else if (slow.kind === 'drop') {
      last = fade(slow, second);
      badge(slow, last);
    } else if (slow.kind === 'arrive') {
      const went = fly(slow, second);
      if (went) {
        last = went.landed;
        // The faster piece, standing where the slower one arrives, is taken there (§2.5 C).
        if (fell) {
          busy = Math.max(busy, strike(svg, fell.node, ...at(fast.to), cell,
                                       { at: went.impact, dx: went.dx, dy: went.dy }));
        }
      }
      badge(slow, last);
    } else {
      badge(slow, second);
    }
  }
  busy = Math.max(busy, last);
  // Every other piece taken off: struck when the move that took it lands, or, with no move
  // ending there (a blast, a piece reborn elsewhere), as the beat nearer to it ends.
  const ex = SESSION.ruleset.explosion;
  const near = square => {
    if (!slow || !fast || slow.kind !== 'arrive') return first;
    const [x, y] = centre(square), [fx, fy] = centre(fast.to), [sx, sy] = centre(slow.to);
    return Math.hypot(sx - x, sy - y) < Math.hypot(fx - x, fy - y) ? last : first;
  };
  for (const square of story.taken) {
    const piece = pieceOf(before[square]);
    if (!piece) continue;
    const [x, y] = at(square);
    const blow = blows.get(square) || { at: near(square), dx: 0, dy: 0 };
    busy = Math.max(busy, ex && !blows.has(square)
      ? Math.max(blastFire(svg, [square], cell, view, blow.at),
                 burn(svg, piece, x, y, cell, blow, null))
      : capture(svg, piece, x, y, cell, blow));
  }
  // The lost move's arrow over every piece and ghost, the one taken under it included.
  if (over) svg.appendChild(over);
  return { busy: busy, landed: last, doomed: new Map() };
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

// The strip of an announced round between two people at two screens (Q935e): the reader's
// part by name, the announced move once it is in (shown to both), the server's sentence of
// the time left and a draw offer. A reader of neither seat is told the phase.
function announcedCorrStatusHtml(number, held) {
  const round = SESSION.round, between = SESSION.correspondence;
  const you = between.you, announcer = round.announcer, shown = round.announced;
  const left = between.toMove ? ` ${esc(between.toMove)}` : '';
  if (you === null || you === undefined) {
    return `${number}${esc(spyLine(SESSION.spy, round, SESSION.sideToMove, between.players))}${left}`;
  }
  const other = esc(between.players[1 - you]);
  const offer = between.drawOffer === null || between.drawOffer === undefined ? ''
    : between.drawOffer === you ? ' You have offered a draw.' : ` ${other} offers a draw.`;
  if (ROUND_SENDING) {
    return shown ? `${number}<b>Your answer is in:</b> waiting for the reveal.`
      : `${number}<b>Your announcement is in.</b>`;
  }
  if (held) {
    return shown
      ? `${number}<b>${esc(held.text)}</b> is your answer to ${esc(shown.text)}, not sent yet.
        You can change it until you answer.`
      : `${number}<b>${esc(held.text)}</b> is your choice, not announced yet. You can change it
        until you announce it.`;
  }
  const note = NOTE ? `<span class="frame-note">${esc(NOTE)}</span>` : '';
  if (you === announcer) {
    return shown
      ? `${number}<b>You announced ${esc(shown.text)}.</b> ${other} answers it.${left}${offer}`
      : `${number}<b>Your move to announce.</b> ${other} holds the only spy and answers knowing
        your move.${left}${offer}${note}`;
  }
  return shown
    ? `${number}<b>${other} announced ${esc(shown.text)}.</b> Your answer is due.${left}${
      offer}${note}`
    : `${number}<b>${other} announces first.</b> You answer once their move is in.${left}${offer}`;
}

// The phase line of a turn of a spy game between two people (Q935e), the players named by
// their handles; nothing in a round or in any other game.
function corrSpyNoteHtml() {
  const spy = SESSION && SESSION.spy;
  const between = SESSION && SESSION.correspondence;
  if (!spy || !between || SESSION.round) return '';
  return `<span class="frame-note" data-spy="${esc(spy.phase)}">${
    esc(spyLine(spy, null, SESSION.sideToMove, between.players))}</span>`;
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
