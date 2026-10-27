const { LINES, SHOWN, SHOWCASE } = JSON.parse(document.getElementById('page-data').textContent);
// What each game is called, by hash. Built from the payload's own game rows, which
// cover every rated game, so *still open* can send hashes alone and every one of them
// resolves here.
let NAMES = {};

// The games whose rules were deleted. Sent as hashes rather than worked out from a name,
// because *deleted game* is a name a game could in principle be given.
let DELETED = new Set();

// UP2: `FLIPPED`, `PIECE_ART`, `esc` and `stillNode` are `still.js`'s, loaded before this.
function el(tag, className, said) {
  const it = document.createElement(tag);
  if (className) it.className = className;
  if (said !== undefined) it.textContent = said;
  return it;
}

function cell(row, text, className) {
  const td = document.createElement('td');
  td.textContent = text;
  if (className) td.className = className;
  row.appendChild(td);
  return td;
}
function count(label, value) {
  const box = document.createElement('div');
  const number = document.createElement('b');
  number.textContent = String(value);
  const name = document.createElement('span');
  name.textContent = label;
  box.appendChild(number);
  box.appendChild(name);
  return box;
}
function span(parent, words, className) {
  const it = document.createElement('span');
  it.textContent = words;
  if (className) it.className = className;
  parent.appendChild(it);
  return it;
}
function bold(parent, said) {
  const it = document.createElement('b');
  it.textContent = said;
  parent.appendChild(it);
  return it;
}
function words(parent, said) { parent.appendChild(document.createTextNode(said)); }
// Q760: a person named on this page, linked to their page so every name is a way to find
// them; somebody who left (*a former player*) has no page and reads as words.
function person(parent, handle) {
  const name = String(handle || '');
  if (!name || name === 'a former player' || !/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(name)) {
    words(parent, name);
    return;
  }
  const it = document.createElement('a');
  it.href = '/players/' + encodeURIComponent(name);
  it.textContent = name;
  parent.appendChild(it);
}
// A deleted game has no page to go to, so its name is text rather than a link. Your
// rows, your milestones and your badges on it stay where they are and read *deleted
// game*, which is the name the store now holds for it.
function gameLink(hash, name) {
  if (DELETED.has(hash)) {
    const said = document.createElement('span');
    said.className = 'state';
    said.textContent = name || 'deleted game';
    return said;
  }
  const link = document.createElement('a');
  link.href = GAME + hash;
  link.textContent = name || UNNAMED;
  return link;
}
// S1: the strongest computer player beaten, by the name the server gives it and the
// rating it held going into that game, on the scale people are rated on.
function rung(game) {
  if (!game.rung) return 'none yet';
  const rating = Math.round(game.rung.rating);
  return (game.rung.label || 'the computer') + ' (rated ' + rating + ' here)';
}
// M4. *none of 5* rather than *0 of 5*, the way the rung beside it says *none yet*.
// An untried game has reached no rung, which is not a nought it scored.
function rungs(game) {
  if (!game.rungs) return '\u2014';
  return (game.rungs.beaten || 'none') + ' of ' + game.rungs.of;
}
// T3, and F8: the person's own rating on the game first, the number they came for, then
// how far it sits from the strongest computer player there. The server sends both
// unrounded and the gap in one direction, below the strongest; which way round to word
// it is the page's. A rating still settling says so, because the game's own table does
// not rank one.
function distance(game) {
  if (!game.distance) return '\u2014';
  const points = Math.round(game.distance.points);
  const gap = points > 0 ? points + ' below the strongest'
    : points < 0 ? (-points) + ' above the strongest' : 'equal to the strongest';
  const said = Math.round(game.distance.rating) + ', ' + gap;
  return game.distance.established ? said : said + ' (still settling)';
}
// S12: a game with no name is *an unnamed game*, never twelve characters of its hash.
const UNNAMED = 'an unnamed game';
function named(game) { return game.name || UNNAMED; }
function nameOf(hash) { return NAMES[hash] || UNNAMED; }
// UP0: a day is written one way on every page that is a member's, *24 Oct 2026*, as the
// member's page (`pages._day`) writes it. Read off the stamp's own date, never through
// the browser's clock, so the day the server stored is the day shown wherever it is read.
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
  'Dec'];
function day(stamp) {
  const held = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(stamp || ''));
  if (!held) return stamp ? String(stamp).slice(0, 10) : '';
  return Number(held[3]) + ' ' + MONTHS[Number(held[2]) - 1] + ' ' + held[1];
}
function seatWord(seat) { return seat === 0 ? 'playing first' : 'playing second'; }

// The games you have actually played, then the rest, each part by name. The payload
// arrives in store order, which put the game somebody played this morning wherever the
// library happened to load it (D2).
function inOrder(games) {
  return games.slice().sort((one, two) => {
    const played = (two.played > 0) - (one.played > 0);
    // Two unnamed games read alike, so their hashes, never shown, keep the order fixed.
    return played || named(one).localeCompare(named(two))
      || String(one.hash).localeCompare(String(two.hash));
  });
}

// The same three facts twice, and CSS shows one set. They are three columns on a desk
// and three lines under the game's name below 760px, where eight columns do not fit.
function narrow(parent, label, said) {
  span(parent, label + ': ' + said, 'narrow');
}

// The counts are absent until the first of them is non-zero: three noughts read as a
// score nobody has started (L2 of `docs/archive/plan-launch.md` §3.2). `drawStart` says what to
// do in their place.
function drawCounts(counts) {
  const where = document.getElementById('counts');
  where.replaceChildren();
  if (!counts.tried) return;
  // S2 and UP0: every tile is absent at nought rather than shown as one, because nobody
  // is behind for winning, beating the strongest or topping no game's table yet (§3,
  // test 5). *games tried* is never nought here: the counts are drawn from the first.
  const tiles = [['games tried', counts.tried], ['games won', counts.won],
    ['strongest beaten', counts.top], ['games where you are top rated', counts.highest]];
  for (const [label, value] of tiles) {
    if (value) where.appendChild(count(label, value));
  }
}

// L2 and UP2, a page with nothing on it yet: the site's key idea first (*invite people to
// make games*, `plan-landing.md`), so *Start building a game* is the one primary, *Play
// today's game* beside it (the front page's own pick, `payload.today`, else the games),
// and one line on what the page will hold. Not the library again by kind, and nothing that
// counts down or counts what the person has not done. Absent once anything is here: a
// game finished, a game made, or a game waiting.
const START = '/game';
const BUILD = '/#/design';
const BUILD_WORDS = "Start building a game";
const TODAY_WORDS = "Play today's game";
function drawStart(counts, rows, busy = false, today = null) {
  const where = document.getElementById('start');
  where.replaceChildren();
  if (counts.tried || busy) return;
  where.appendChild(el('p', 'lead', 'Nothing here yet.'));
  const doors = el('div', 'startDoors');
  const build = el('a', 'button primary', BUILD_WORDS);
  build.href = BUILD;
  doors.appendChild(build);
  const play = el('a', 'button', TODAY_WORDS);
  play.href = today && today.hash ? GAME + today.hash : START;
  doors.appendChild(play);
  where.appendChild(doors);
  where.appendChild(el('p', 'under', 'This page will hold the games you make, each on its '
    + 'way from a draft to the library; the games waiting for your move; and the rated '
    + 'games you have played, with what you have done on them.'));
}

// The table is hidden with the counts: before a first finished game every row of it
// would read *none yet*. UP0: a row is a game you have played, never one you have not;
// the rest are one line under the table (`drawUntried`), not ninety rows of noughts.
function drawGames(rows, started = true) {
  document.getElementById('record').hidden = !started;
  document.getElementById('record-under').hidden = !started;
  const games = document.getElementById('games');
  games.replaceChildren();
  for (const game of inOrder(rows.filter(game => game.played > 0))) {
    const row = document.createElement('tr');
    const first = cell(row, '');
    first.appendChild(gameLink(game.hash, named(game)));
    // Your games stay when the rules leave the corpus; the mark says why the library
    // no longer offers this one. A deleted game says so in the name `gameLink` drew and
    // takes no mark of its own.
    if (game.retired && !DELETED.has(game.hash)) span(first, ' retired', 'state retired');
    narrow(first, 'best beaten', rung(game));
    narrow(first, 'computer players beaten', rungs(game));
    narrow(first, 'your rating', distance(game));
    cell(row, game.played, 'num');
    cell(row, game.won, 'num');
    cell(row, game.drawn, 'num');
    cell(row, game.lost, 'num');
    cell(row, rung(game), 'best');
    cell(row, rungs(game), 'beat num');
    cell(row, distance(game), 'dist num');
    games.appendChild(row);
  }
}

// UP0: the rated games you have not played, as one line under the table: how many, and
// how many of those nobody has beaten, each linked to the library with its filter set
// (`/game?tag=`, read once by `browse.js`). The same games `drawStart` offers by name to
// a member with nothing played: in the library now, never retired or deleted. Absent
// when there are none, and the second clause absent when it would be a nought.
const UNTRIED_LINK = '/game?tag=untried';
const UNBEATEN_LINK = '/game?tag=unbeaten';
function drawUntried(rows, started = true) {
  const where = document.getElementById('untried');
  if (!where) return;
  where.replaceChildren();
  if (!started) return;
  const open = rows.filter(game => !game.played && !game.retired && !DELETED.has(game.hash));
  if (!open.length) return;
  const unbeaten = open.filter(game => game.unbeaten === true).length;
  const link = (href, said) => {
    const it = document.createElement('a');
    it.href = href;
    it.textContent = said;
    where.appendChild(it);
  };
  words(where, 'You have not played ' + open.length + ' of the rated games yet ('
    + (open.length === 1 ? 'it is' : 'they are') + ' under ');
  link(UNTRIED_LINK, 'untried by you');
  words(where, ' in the library)');
  if (unbeaten) {
    words(where, (open.length === 1 ? ', and nobody has beaten the strongest computer '
      + 'player on it yet' : ', and on ' + unbeaten + ' of those nobody has beaten the '
      + 'strongest computer player yet') + ' (every such game is under ');
    link(UNBEATEN_LINK, 'Unbeaten');
    words(where, ')');
  }
  words(where, '.');
}

// B2. One rated game you left running, as a line. The name is the link, and it goes to
// the board rather than to the game's page: the point of the line is the game in
// progress, and the game's page is one click on from it.
//
// An entry whose rules the store will not hand back -- a deleted game, or one taken off
// the site -- draws no link at all. There is nothing to replay the moves through, so no
// board can be drawn, and the sentence says so rather than offering a link that refuses.
//
// UP2: a row of *Waiting on you*, with the position as the engine left it (the server's
// `position`, its last move tinted) and one action, *Carry on*.
function carryItem(entry) {
  const name = entry.name || nameOf(entry.game);
  let title;
  if (entry.openable) {
    title = el('a', '', name);
    title.href = PLAY + entry.id;
  } else {
    title = el('b', '', name);
  }
  const row = waitRow('Left running \u00b7 rated, against the computer', title,
    entry.openable ? waitingBoard(entry.position, entry.game, entry.lastMove, moveKey(entry)) : null);
  // S8: a named opponent always (the server's `label`), and turns, each player's move
  // being one.
  words(row.facts, 'Against ' + (entry.label || 'the computer') + ', playing '
    + (entry.seat === 0 ? 'first' : 'second') + ', ' + entry.plies + ' turn'
    + (entry.plies === 1 ? '' : 's') + ' played'
    + (entry.moved ? ', last moved ' + day(entry.moved) : ', not moved yet') + '.');
  if (!entry.openable) {
    span(row.facts, 'These rules are gone, so the board cannot be drawn. The game stays open '
      + 'until it is resigned after thirty days without a move.', 'rule');
  } else {
    const go = el('a', 'button main', 'Carry on');
    go.href = PLAY + entry.id;
    row.acts.appendChild(go);
  }
  return row.item;
}

// UP5, `docs/plan-user-page.md` §5: the user page's one motion. A waiting game's board, its
// last move (the engine's `lastMove`, the session's own history entry) played once, the
// first time the board comes into view: the piece slides from where it stood to where it
// landed (`travel`), or a piece from a hand drops in (`stoneDrop`), with the board motion
// every page shares (`motion.js`). Nothing else on the page moves: a capture's blow, a
// portal's journey and a round's reveal are left to the tint, which says them already. Under
// `prefers-reduced-motion` nothing moves at all, asked at the moment the board is seen. Where
// a square is drawn is `motion.js`'s to ask the drawing (`stillMove`); which squares the move
// joined is the engine's.
const SEEN = new Set();  // the moves already played on this page, so a redraw does not replay one
function stillMoving() {
  try { return !window.matchMedia('(prefers-reduced-motion: reduce)').matches; }
  catch (e) { return false; }
}
function waitingBoard(still, hash, last, key) {
  const plays = !!last && last.to !== null && last.to !== undefined && last.portal === undefined
    && last.foldIn === undefined && !last.reveal && !last.pass;
  const box = stillNode(still, hash, last, false, plays);
  if (plays && key !== undefined) slideOnSight(box, still, last, key);
  return box;
}
function slideOnSight(box, still, last, key) {
  if (SEEN.has(key) || typeof stillMove !== 'function' || typeof IntersectionObserver !== 'function'
      || !stillMoving()) return;
  const sight = new IntersectionObserver(seen => {
    if (!seen.some(one => one.isIntersecting)) return;
    sight.disconnect();
    if (SEEN.has(key) || !stillMoving()) return;
    SEEN.add(key);
    stillMove(box.querySelector('svg'), still, last);
  }, { threshold: 0.6 });
  sight.observe(box);
}

// UP5: which move a waiting row's board plays once (`still.js` `waitingBoard`): the game and
// its last move, so a redraw of the row after an answer does not play the same move again.
function moveKey(entry) {
  return entry.id + ':' + JSON.stringify(entry.lastMove || null);
}

// UP2: one row of *Waiting on you*: a small board, what is waiting (`kicker`), the game's
// name (`title`, a node), a line of facts the caller fills, and its actions. `yours` rings
// a game where it is your move, the one thing on the page that needs you most.
function waitRow(kicker, title, board, yours = false) {
  const item = el('li', 'waitRow' + (yours ? ' yours' : ''));
  item.appendChild(board || el('div', 'stillBox'));
  const text = el('div');
  text.appendChild(el('div', 'kicker', kicker));
  const name = el('div', 'title');
  name.appendChild(title);
  text.appendChild(name);
  const facts = el('p', 'facts');
  text.appendChild(facts);
  item.appendChild(text);
  const acts = el('div', 'acts');
  item.appendChild(acts);
  return { item, facts, acts };
}

// The rated games left running, newest first, which is the order the server sends, as rows
// of *Waiting on you*. Absent when nothing is running, because an empty list would read as
// a nought. The rule is said once, under the rows (`drawQuiet`).
const IDLE_RULE = 'A rated game against the computer is resigned after thirty idle days, '
  + 'or when you start another rated game on the same game.';
function drawCarryOn(entries) {
  const where = document.getElementById('carry');
  where.replaceChildren();
  if (!entries.length) return;
  const list = el('ul', 'waitRows');
  for (const entry of entries) list.appendChild(carryItem(entry));
  where.appendChild(list);
}

// Q570, Q627, Q572. *Correspondence* (s6 of the mock): the line of what you finished
// (`docs/archive/plan-return.md` §1.6), absent until there is one, then your running games
// against other people under *Your move* and *Their move*, in the server's order
// (`correspondence.started_order`: your move first, then theirs, each by the time left to
// move). When the last move landed and the time left to move are facts in the server's
// words (the dated exception to test 2): nothing here counts down, no count in a heading,
// no colour that alarms. The rule is said once, as a rule. Each part is absent when it
// has nothing in it; the section's heading stays for the email box under it.
//
// UP2: *Your move* is a row of *Waiting on you*, with the position as it stands and the
// opponent's last move tinted, and *Make your move*; *Their move* is a quiet line under the
// rows (`#between-quiet`), with the finished line and the rule said once.
function drawBetween(between) {
  const where = document.getElementById('between');
  const quiet = document.getElementById('between-quiet');
  if (!where || !quiet) return;
  where.replaceChildren();
  quiet.replaceChildren();
  const running = (between && between.running) || [];
  const finished = between && between.finished;
  if (running.length) {
    const mine = running.filter(entry => entry.yourMove);
    const rows = el('ul', 'waitRows');
    for (const entry of mine) {
      const name = el('a', '', entry.name || nameOf(entry.game));
      name.href = PLAY + entry.id;
      const row = waitRow('Your move \u00b7 by correspondence', name,
        waitingBoard(entry.position, entry.game, entry.lastMove, moveKey(entry)), true);
      opponentLine(row.facts, entry);
      const go = el('a', 'button main', 'Make your move');
      go.href = PLAY + entry.id;
      row.acts.appendChild(go);
      rows.appendChild(row.item);
    }
    if (mine.length) where.appendChild(rows);
    const lines = el('ul', 'quiet');
    for (const entry of running.filter(entry => !entry.yourMove)) {
      const item = el('li');
      bold(item, 'Their move');
      const link = el('a', '', entry.name || nameOf(entry.game));
      link.href = PLAY + entry.id;
      item.appendChild(link);
      words(item, ' ');
      opponentLine(item, entry);
      lines.appendChild(item);
    }
    if (lines.children.length) quiet.appendChild(lines);
    quiet.appendChild(el('p', 'under', 'No clock: each of you moves when you like. '
      + between.rule));
  }
  if (finished) quiet.appendChild(el('p', 'under', finished));
}

// Who a game between people is against, with their mark (R2), and the server's facts:
// rated or practice, when the last move landed, and the time to move in its own words.
function opponentLine(where, entry) {
  words(where, 'against ');
  const mark = window.memberMark && window.memberMark(entry.opponentMark, 18);
  if (mark) where.appendChild(mark);
  person(where, entry.opponent);
  words(where, ' \u00b7 ' + (entry.rated ? 'rated' : 'practice')
    + ', last moved ' + day(entry.moved) + '. ');
  if (entry.toMove) words(where, entry.toMove);
}

// Q572, `docs/archive/plan-return.md` §1.5 and s6 of the mock. The one email's box, off unless
// the person ticks it, with the server's sentences saying exactly what it does and that
// nothing else is ever sent. Ticking or unticking is saved at once and said in the note.
function drawMoveEmail(box) {
  const where = document.getElementById('move-email');
  if (!where || !box) return;
  where.replaceChildren();
  const heading = document.createElement('h3');
  heading.textContent = 'Email';
  where.appendChild(heading);
  const label = document.createElement('label');
  label.className = 'u-d-block';
  const tick = document.createElement('input');
  tick.type = 'checkbox';
  tick.id = 'move-email-box';
  tick.checked = !!box.on;
  label.appendChild(tick);
  words(label, ' ' + box.said);
  where.appendChild(label);
  const under = document.createElement('p');
  under.className = 'under';
  under.textContent = box.under;
  where.appendChild(under);
  tick.addEventListener('change', () => {
    tick.disabled = true;
    post('/api/move-email', { on: tick.checked })
      .then(fresh => {
        drawMoveEmail(fresh);
        say(fresh.on ? 'Move emails are on.' : 'Move emails are off.');
      })
      .catch(error => { tick.checked = !tick.checked; tick.disabled = false; say(error.message); });
  });
}

// Q719: *Invite somebody*. How many invitations the site left this person to send, as a
// plain count, and the address box while there is one to send; at zero the count's own
// sentence says so and the box goes. Absent for somebody who was never given any. The
// person types an address and nothing else: the invitation is the site's own.
function drawInvite(box) {
  const where = document.getElementById('invite');
  if (!where) return;
  where.replaceChildren();
  if (!box) return;
  const heading = document.createElement('h3');
  heading.textContent = 'Invite somebody';
  where.appendChild(heading);
  const count = document.createElement('p');
  count.id = 'invite-left';
  count.textContent = box.said;
  where.appendChild(count);
  if (box.left > 0) {
    const form = document.createElement('form');
    form.className = 'invite';
    const label = document.createElement('label');
    const name = document.createElement('span');
    name.textContent = 'their email address';
    const field = document.createElement('input');
    field.type = 'email';
    field.id = 'invite-email';
    field.required = true;
    field.autocomplete = 'off';
    field.setAttribute('autocapitalize', 'off');
    field.spellcheck = false;
    label.append(name, field);
    const send = document.createElement('button');
    send.type = 'submit';
    send.textContent = 'send the invitation';
    form.append(label, send);
    where.appendChild(form);
    const under = document.createElement('p');
    under.className = 'under';
    under.textContent = box.under;
    where.appendChild(under);
    form.addEventListener('submit', event => {
      event.preventDefault();
      send.disabled = true;
      post('/api/invite', { email: field.value })
        .then(answer => { drawInvite(answer.box); say(answer.said); })
        .catch(error => { send.disabled = false; say(error.message); });
    });
  }
}

// Q606, `docs/archive/plan-return.md` §3 C4 (R10): *list me on the games I made as taking
// challengers*, one switch, off unless you switch it on, and absent for somebody with no
// published game of their own. Saved at once and said in the note, as the boxes above are.
function drawTakesChallengers(box) {
  const where = document.getElementById('takes-challengers');
  if (!where) return;
  where.replaceChildren();
  if (!box || !(box.games || []).length) return;
  where.appendChild(el('h3', '', 'Challengers'));
  const label = document.createElement('label');
  label.className = 'u-d-block';
  const tick = document.createElement('input');
  tick.type = 'checkbox';
  tick.id = 'takes-challengers-box';
  tick.checked = !!box.on;
  label.appendChild(tick);
  words(label, ' ' + box.said);
  where.appendChild(label);
  const under = document.createElement('p');
  under.className = 'under';
  under.textContent = box.under + ' Your games: '
    + box.games.map(game => game.name).join(', ') + '.';
  where.appendChild(under);
  tick.addEventListener('change', () => {
    tick.disabled = true;
    post('/api/takes-challengers', { on: tick.checked })
      .then(fresh => {
        drawTakesChallengers(fresh);
        say(fresh.on ? 'You are listed on your games as taking challengers.'
          : 'You are no longer listed on your games as taking challengers.');
      })
      .catch(error => { tick.checked = !tick.checked; tick.disabled = false; say(error.message); });
  });
}

// Q596, `docs/archive/plan-return.md` §6.2 (R1): your page as other members see it, and its one
// box, *show my five-star games on my page*, off unless you tick it. Saved at once and
// said in the note, as the email box is.
function drawYourPage(address, box) {
  const where = document.getElementById('your-page');
  if (!where || !address) return;
  where.replaceChildren();
  const heading = document.createElement('h3');
  heading.textContent = 'Your page';
  where.appendChild(heading);
  const lead = document.createElement('p');
  lead.className = 'under';
  words(lead, 'Other members, and nobody else, see ');
  const link = document.createElement('a');
  link.href = address;
  link.textContent = 'your page';
  lead.appendChild(link);
  words(lead, ': your mark and handle, when you joined, what you have finished lately, '
    + 'the games you designed that are published, your milestones and badges, and who '
    + 'you play most. Visitors cannot see it.');
  where.appendChild(lead);
  if (!box) return;
  const label = document.createElement('label');
  label.className = 'u-d-block';
  const tick = document.createElement('input');
  tick.type = 'checkbox';
  tick.id = 'favourites-box';
  tick.checked = !!box.on;
  label.appendChild(tick);
  words(label, ' ' + box.said);
  where.appendChild(label);
  const under = document.createElement('p');
  under.className = 'under';
  under.textContent = box.under;
  where.appendChild(under);
  tick.addEventListener('change', () => {
    tick.disabled = true;
    post('/api/favourites', { on: tick.checked })
      .then(fresh => {
        drawYourPage(address, fresh);
        say(fresh.on ? 'Your five-star games are on your page.'
          : 'Your five-star games are off your page.');
      })
      .catch(error => { tick.checked = !tick.checked; tick.disabled = false; say(error.message); });
  });
}

// Q571, `docs/archive/plan-return.md` §1.4 and s6 of the mock. *A challenge for you* with accept
// and decline, *Challenges you sent* with withdraw (a link with its address to copy
// again), and *Open to a game of* with remove. Each part is absent when it has nothing
// in it, and nothing here counts down: a challenge waits until it is answered (test 2).
// A declined challenge says so and no more.
function drawChallenges(asked) {
  const where = document.getElementById('challenges');
  const quiet = document.getElementById('challenges-quiet');
  if (!where || !quiet) return;
  where.replaceChildren();
  quiet.replaceChildren();
  const forYou = (asked && asked.forYou) || [];
  const sent = (asked && asked.sent) || [];
  const open = (asked && asked.open) || [];
  if (!forYou.length && !sent.length && !open.length) return;
  const act = (path, body) => post(path, body).then(fresh => {
    if (fresh && fresh.id) { window.location.href = PLAY + fresh.id; return; }
    drawChallenges(fresh);
  }).catch(error => say(error.message));
  const heading = (text) => {
    const it = document.createElement('h3');
    it.textContent = text;
    quiet.appendChild(it);
  };
  const button = (parent, text, kind, press) => {
    const it = document.createElement('button');
    it.textContent = text;
    if (kind) it.className = kind;
    it.addEventListener('click', press);
    parent.appendChild(it);
  };
  if (asked.pausedSays) {
    const line = document.createElement('p');
    line.className = 'under';
    line.textContent = asked.pausedSays;
    quiet.appendChild(line);
  }
  // UP2: each challenge for you is a row of *Waiting on you*, the game's opening as its
  // board, *Accept* its action and *decline* beside it.
  if (forYou.length) {
    const list = el('ul', 'waitRows');
    for (const entry of forYou) {
      const row = waitRow('A challenge for you', gameLink(entry.game, entry.name),
        waitingBoard(entry.position, entry.game, entry.lastMove));
      const mark = window.memberMark && window.memberMark(entry.fromMark, 18);
      if (mark) row.facts.appendChild(mark);
      person(bold(row.facts, ''), entry.from);
      words(row.facts, entry.rematch ? ' offers a rematch' : ' challenged you');
      words(row.facts, ' · ' + entry.seat + ' · ' + (entry.rated ? 'rated' : 'practice')
        + ' · sent ' + day(entry.made) + '.');
      button(row.acts, 'Accept', 'main', () =>
        act('/api/challenge/' + entry.id + '/accept', {}));
      button(row.acts, 'decline', '', () => act('/api/challenge/' + entry.id + '/decline', {}));
      list.appendChild(row.item);
    }
    where.appendChild(list);
  }
  if (sent.length) {
    heading('Challenges you sent');
    const list = document.createElement('ul');
    list.className = 'lines';
    for (const entry of sent) {
      const item = document.createElement('li');
      if (entry.to) { person(item, entry.to); words(item, ', to '); }
      else words(item, 'a link, to ');
      item.appendChild(gameLink(entry.game, entry.name));
      if (entry.answer === 'declined') {
        span(item, ' · declined', 'muted small');
      } else {
        span(item, ' · ' + (entry.to ? 'sent ' + day(entry.made) : 'not used yet') + ' ',
          'muted small');
        if (entry.link) {
          const address = window.location.origin + entry.link;
          button(item, 'copy', '', () => {
            if (navigator.clipboard) navigator.clipboard.writeText(address).catch(() => {});
            say(address);
          });
        }
        button(item, 'withdraw', '', () =>
          act('/api/challenge/' + entry.id + '/withdraw', {}));
      }
      list.appendChild(item);
    }
    quiet.appendChild(list);
  }
  if (open.length) {
    heading('Open to a game of');
    const line = document.createElement('p');
    open.forEach((entry, index) => {
      if (index) words(line, ' · ');
      line.appendChild(gameLink(entry.game, entry.name));
      words(line, ' ');
      button(line, 'remove', '', () => post('/api/game/' + entry.game + '/open', { on: false })
        .then(() => draw()).catch(error => say(error.message)));
    });
    span(line, ' (tick the box on any game\'s page to add one)', 'muted small');
    quiet.appendChild(line);
  }
}

// Q503b, §3.4, and UP2 (direction B, `docs/plan-user-page.md` §3.2): *Your workshop*, the
// games this person made on the path every game takes. The columns are the server's
// (`workshop.columns`: the four levels and *Measured*, a draft whose score the site has
// checked), each game in the column it stands in, with its board, the change from the game
// it was made from in the designer's words, the designer's rungs as dots, and its next step
// in the server's words (`step`). The page holds no level word and decides no column: a
// new level moves the page with no page edit. *Start a new game* opens the designer;
// *Try another change* opens it on a copy of the game a draft was made from, the way a game
// page's *make your own from this* does.
const DESIGN = '/#/design/';
const START_NEW = 'Start a new game';

function drawYourGames(groups, columns = [], lead = false) {
  const where = document.getElementById('mine');
  where.replaceChildren();
  const games = {};
  for (const group of groups) for (const game of group.games) games[game.hash] = game;
  const head = el('div', 'secHead');
  head.appendChild(el('h2', '', 'Your workshop'));
  const start = el('a', 'button' + (lead ? ' primary' : ''), START_NEW);
  start.href = BUILD;
  head.appendChild(start);
  if (!groups.length) {
    // A member who has made nothing yet: the door and one line, never an empty path.
    where.appendChild(head);
    where.appendChild(el('p', 'lead leadIn', 'You have not made a game yet. The designer '
      + 'builds one from a palette of rules, plays it at once and checks it as you edit.'));
    return;
  }
  where.appendChild(head);
  where.appendChild(madeFromLine(Object.values(games)));
  const pipe = el('div', 'pipe' + (columns.length > 5 ? ' extra' : ''));
  columns.forEach((column, index) => {
    const col = el('section', 'pipeCol' + (index ? '' : ' first')
      + (column.games.length || !index ? '' : ' empty'));
    const header = el('header');
    header.appendChild(el('b', '', column.word));
    header.appendChild(el('span', '', column.says));
    col.appendChild(header);
    for (const hash of column.games) if (games[hash]) col.appendChild(miniCard(games[hash]));
    if (!index) col.appendChild(tryAnother(Object.values(games)));
    pipe.appendChild(col);
  });
  where.appendChild(pipe);
}

// The games these were made from, said once: each parent with its level and its score
// with the basis it carries, and the games made from it. *The line of thought* of §3.2.
function madeFromLine(games) {
  const line = el('p', 'lead leadIn');
  const parents = [];
  for (const game of games) {
    const parent = game.parent;
    if (!parent || !parent.name) continue;
    let found = parents.find(one => one.parent.hash === parent.hash);
    if (!found) { found = { parent, names: [] }; parents.push(found); }
    found.names.push(game.name);
  }
  words(line, 'From an idea to the library, one step at a time: each game stands in the '
    + 'column of where it is.');
  for (const { parent, names } of parents) {
    words(line, ' Made from ');
    line.appendChild(gameLink(parent.hash, parent.name));
    const said = [parent.level, scoreWords(parent.score)].filter(Boolean).join(', ');
    words(line, (said ? ' (' + said + ')' : '') + ': ' + listed(names) + '.');
  }
  return line;
}

// A score as the site lets one be said: the number with its run count, and a re-certified
// one says so; a provisional one is marked, with its revision and its date, never bare.
function scoreWords(score) {
  if (!score || score.index === null || score.index === undefined) return '';
  const runs = (score.seeds || []).length || score.totalsUsed || 0;
  const number = Number(score.index).toFixed(3);
  const counted = runs ? ' \u00b7 ' + runs + (runs === 1 ? ' run' : ' runs') : '';
  if (score.status !== 'current') {
    return 'provisional: ' + number + counted + ' (' + [score.revision, day(score.measuredAt)]
      .filter(Boolean).join(', ') + ')';
  }
  return number + counted + (score.certified ? ', re-certified from a sample' : '');
}

function listed(names) {
  if (names.length < 2) return names.join('');
  return names.slice(0, -1).join(', ') + ' and ' + names[names.length - 1];
}

// One game in its column: its board, its name, the change, the rungs and the next step.
function miniCard(game) {
  const card = el('article', 'mini');
  card.appendChild(stillNode(game.still, game.hash));
  const name = el('h3');
  name.appendChild(gameLink(game.hash, game.name));
  card.appendChild(name);
  for (const sentence of game.change || []) card.appendChild(el('p', 'change', sentence));
  if ((game.rungs || []).length) card.appendChild(rungDots(game.rungs));
  stepTo(card, game);
  // Q599: how many voters chose each fixed tag, a count and never who.
  const tagged = game.tagCounts || [];
  if (tagged.length) {
    card.appendChild(el('p', 'facts', 'Tagged: ' + tagged.map(entry => entry.words + ' · '
      + entry.count).join(', ') + '.'));
  }
  if (game.inDesigner) {
    const open = el('a', 'go', 'open in the designer');
    open.href = DESIGN + game.hash + AS_DRAFT;
    card.appendChild(open);
  }
  return card;
}

// UP7 (`docs/reference/user-page-cold-read.md`, findings 1 and 2). Your own game opens in
// the designer as itself (`?as=draft`; the designer asks the server whether it is yours),
// never as a copy; *Try another change* is the one way to a copy. A next step taken at the
// designer's measuring control (`stepAt`, the server's) is a link that opens the draft
// there, with the server's one line on what measuring asks (`stepNote`).
const AS_DRAFT = '?as=draft';
const AT_MEASURE = '&at=measure';

function stepTo(card, game) {
  if (!game.step) return;
  if (game.stepAt !== 'measure') {
    card.appendChild(el('p', 'facts', game.step));
    return;
  }
  const line = el('p', 'facts');
  const link = el('a', 'stepGo', game.step);
  link.href = DESIGN + game.hash + AS_DRAFT + AT_MEASURE;
  line.appendChild(link);
  card.appendChild(line);
  if (game.stepNote) card.appendChild(el('p', 'under', game.stepNote));
}

// The designer's rungs as dots, each named by its title, in the marks the designer draws.
function rungDots(rungs) {
  const row = el('span', 'rungDots');
  for (const rung of rungs) {
    const dot = el('i', rung.state);
    dot.title = rung.name + ': ' + rung.says;
    row.appendChild(dot);
  }
  return row;
}

// *Try another change*: the designer on a copy of the game a draft was made from, one door
// per such game, or on the member's own game where none was made from another.
function tryAnother(games) {
  const door = el('article', 'mini door');
  door.appendChild(el('b', '', 'Try another change'));
  const seen = new Set();
  const from = [];
  for (const game of games) {
    const parent = game.parent && game.parent.name ? game.parent : null;
    const target = parent ? { hash: parent.hash, name: parent.name }
      : { hash: game.hash, name: game.name };
    if (seen.has(target.hash)) continue;
    seen.add(target.hash);
    from.push(target);
  }
  const line = el('p');
  words(line, 'Open the designer on a copy of ');
  from.slice(0, 3).forEach((target, index) => {
    if (index) words(line, index === Math.min(from.length, 3) - 1 ? ' or ' : ', ');
    const link = el('a', '', target.name);
    link.href = DESIGN + target.hash;
    line.appendChild(link);
  });
  words(line, '; it checks the draft as you edit.');
  door.appendChild(line);
  return door;
}

// Q503a. The games a save had listed, which went back to Draft when saving stopped
// listing (`0016_drafts.sql`). Each still opens by its link, so each is a link. Absent
// when there are none, as every section here is.
function drawBackToDraft(games) {
  const where = document.getElementById('drafts');
  where.replaceChildren();
  if (!games.length) return;
  const heading = document.createElement('h2');
  heading.textContent = 'Your games back to Draft';
  where.appendChild(heading);
  const lead = document.createElement('p');
  lead.className = 'under';
  lead.textContent = 'Saving a game used to list it. Since ' + day(games[0].when)
    + ' a save keeps a draft, and a game is listed only when its designer publishes it '
    + 'with a score measured on their own computer and checked by the site. These '
    + 'games had no such score, so they went back to Draft. Their links still work, and '
    + 'nothing else about them changed. To publish one, open it, measure its score on '
    + 'your computer from its page, and once the site has checked that score, a few '
    + 'minutes later, press Publish.';
  where.appendChild(lead);
  const list = document.createElement('ul');
  list.className = 'lines';
  for (const game of games) {
    const item = document.createElement('li');
    item.appendChild(gameLink(game.hash, game.name));
    list.appendChild(item);
  }
  where.appendChild(list);
}

// One milestone, as a line. The rule sentence goes under it, because a page that named
// a rule only once you had met it would be hiding every rule you had not (§3, test 7).
// UP2: milestones of one kind on one day are one line naming every game (`more`), the
// rule said once; such a line names no seat, which differs from game to game.
function milestoneItem(line, more = []) {
  const item = document.createElement('li');
  const said = LINES[line.kind] || [line.kind, 'on'];
  bold(item, said[0]);
  words(item, ', ' + said[1] + ' ');
  const all = [line, ...more];
  all.forEach((one, index) => {
    if (index) words(item, index === all.length - 1 ? ' and ' : ', ');
    item.appendChild(gameLink(one.game, one.name));
  });
  // A designer's line (D1 to D4) has no seat; D4 says who forked it.
  if (line.by) { words(item, ', by '); person(item, line.by); }
  const seat = !more.length && (line.seat === 0 || line.seat === 1)
    ? ', ' + seatWord(line.seat) : '';
  words(item, ', ' + day(line.when) + seat + '.');
  span(item, line.rule, 'rule');
  return item;
}

// Milestones of one kind on one day, by the same person where one is named, as one group,
// newest first as the lines arrive sorted below.
function grouped(lines) {
  const groups = [];
  for (const line of lines) {
    const found = groups.find(group => group[0].kind === line.kind
      && day(group[0].when) === day(line.when) && (group[0].by || '') === (line.by || ''));
    if (found) found.push(line); else groups.push([line]);
  }
  return groups;
}

// One badge, as a line among the milestones (box E4 of `docs/archive/plan-meta-game.md`).
// The name, the sentence and the rule are the server's, out of the one table in
// `site/badges.py`, and the figures are the ones the row kept on the day. Nothing here is
// a total: only badges you hold arrive, and the page neither counts them nor says how
// close anything else is.
function badgeItem(badge) {
  const item = document.createElement('li');
  bold(item, badge.name);
  if (badge.game) {
    words(item, ', on ');
    item.appendChild(gameLink(badge.game, badge.name_of));
  }
  words(item, ', ' + day(badge.when) + '. ' + badge.sentence);
  // A score or a star quotes what stood that day, under the weights named beside it,
  // and is compared with nothing since -- so it reads as history and not as a standing.
  if (badge.quoted) words(item, ' On that day, ' + badge.quoted + '.');
  span(item, badge.rule, 'rule');
  return item;
}

// The dated lines, newest first. Milestones and badges share one list, which is what
// makes the date the thing you read down rather than the kind of record.
function drawDone(lines, held) {
  const where = document.getElementById('done');
  where.replaceChildren();
  const items = [];
  const newest = lines.slice().sort((one, two) => String(two.when || '').localeCompare(
    String(one.when || '')));
  for (const [line, ...more] of grouped(newest)) {
    items.push([line.when || '', milestoneItem(line, more)]);
  }
  for (const badge of held) items.push([badge.when || '', badgeItem(badge)]);
  if (!items.length) return;
  items.sort((one, two) => String(two[0]).localeCompare(String(one[0])));
  const heading = document.createElement('h2');
  heading.textContent = 'What you have done';
  where.appendChild(heading);
  const list = document.createElement('ul');
  list.className = 'lines';
  for (const [, item] of items) list.appendChild(item);
  where.appendChild(list);
}

// UP3 (choice C1): *Lately*, the server's lines (`site/lately.py`), dated and newest first
// as they arrive. Each sentence is the server's, in order: words, a game (linked as every
// game on this page is) or a person (linked to their page). Nothing is marked unread,
// counted or kept: the page reads no last visit and writes none. Absent with no line.
// The newest `LATE_SHOWN` lead; the rest are under *Earlier*, folded on a phone (where the
// list stands above the record and every line is a screen's worth of scrolling) and open on
// a desk, where it stands beside the record.
const LATE_SHOWN = 2;
function wide() {
  try { return !!window.matchMedia && window.matchMedia('(min-width: 1101px)').matches; }
  catch (error) { return false; }
}
function drawLately(lines) {
  const section = document.getElementById('lately');
  const lead = document.getElementById('late-lines');
  const more = document.getElementById('late-more');
  const earlier = document.getElementById('late-earlier');
  if (!section || !lead) return;
  lead.replaceChildren();
  if (earlier) earlier.replaceChildren();
  lines.forEach((line, index) => {
    const where = index < LATE_SHOWN || !earlier ? lead : earlier;
    const item = el('li');
    item.appendChild(el('span', 'when', day(line.when)));
    const said = el('span');
    for (const part of line.parts || []) {
      if (typeof part === 'string') words(said, part);
      else if (part.game !== undefined) said.appendChild(gameLink(part.game, part.name));
      else if (part.person !== undefined) person(said, part.person);
    }
    item.appendChild(said);
    where.appendChild(item);
  });
  if (more) {
    more.hidden = lines.length <= LATE_SHOWN;
    more.open = wide();
  }
  section.hidden = !lines.length;
}

// R9b (`site/proven.py`): the proven positions this person solved, theirs alone. A count
// of their own marks against what the game offers, never anybody else's, and nothing at
// all until there is a mark (§1.5: no empty heading). Since Q993b each game names its share
// solved and links to its positions' own page, and the share across every game with
// proven positions leads, counted live by the server. UP2: one line under the record,
// its rule under it.
function drawProven(lines, share) {
  const where = document.getElementById('proven');
  if (!where) return;
  where.replaceChildren();
  if (!lines.length) return;
  const line = el('p');
  line.id = 'provenShare';
  bold(line, 'Proven positions. ');
  if (share && share.said) words(line, share.said + ' ');
  lines.forEach((one, index) => {
    if (index) words(line, '; ');
    line.appendChild(provenLink(one.hash, one.name));
    const said = [];
    if (one.of !== null && one.of !== undefined) {
      said.push('solved ' + one.solved + ' of ' + one.of + ' (' + one.share + ')');
    }
    said.push(one.of === null ? 'converted ' + one.converted
      : 'converted ' + one.converted + ' of ' + one.of);
    if (one.found || one.singles) {
      said.push(one.singles === null ? 'found the move in ' + one.found
        : 'found the move in ' + one.found + ' of ' + one.singles);
    }
    words(line, ': ' + said.join(', '));
  });
  words(line, '.');
  where.appendChild(line);
  where.appendChild(el('span', 'rule', 'Positions the computer proved won for the side to '
    + "move, each game's on a page of its own. Only you see this. A try that did not end in "
    + 'your win is not counted against you, and nothing here expires. Each count is of the '
    + 'positions the game offers now, so one later left out as too easy is no longer '
    + 'counted, and one added lowers the share.'));
}

// A game's proven positions, at their own address (Q993b): the game's name, linked to the
// game's own address (`gameLink`, the one place it is written) and on to its positions.
function provenLink(hash, name) {
  const link = gameLink(hash, name);
  if (link.tagName === 'A') link.setAttribute('href', link.getAttribute('href') + PROVEN);
  return link;
}
const PROVEN = '/proven';

// UP2 (§3.2): beside what waits on you, the member's game furthest along the path, large:
// its board and hand, the four levels with its own lit (`stages.track`), its next step in
// the server's words. The furthest is the first game in the last column the server marks
// with a `feature` (its kicker, `workshop.FEATURED`) that holds one; no level is read here.
function drawFeatured(groups, columns) {
  const where = document.getElementById('featured');
  if (!where) return null;
  where.replaceChildren();
  const games = {};
  for (const group of groups) for (const game of group.games) games[game.hash] = game;
  let shown = null, kicker = '';
  for (const column of columns) {
    if (column.feature && column.games.length && games[column.games[0]]) {
      shown = games[column.games[0]];
      kicker = column.feature;
    }
  }
  if (!shown) return null;
  const card = el('figure', 'feature');
  card.appendChild(el('div', 'kicker', kicker));
  const name = el('h3');
  name.appendChild(gameLink(shown.hash, shown.name));
  card.appendChild(name);
  card.appendChild(stillNode(shown.still, shown.hash, null, true));
  card.appendChild(trackList(shown.track || []));
  stepTo(card, shown);
  where.appendChild(card);
  return shown;
}

// The four levels with this game's lit, in the application's own markup (`core.js`
// `trackHtml`, the `.track` rule), from the server's `track`.
function trackList(track) {
  const list = el('ol', 'track');
  for (const step of track) list.appendChild(el('li', step.lit ? 'lit' : '', step.word));
  return list;
}

// *Waiting on you*: its heading over the rows, the rule for games left running said once
// under them, and the first row's action as the page's one primary. With nothing that
// needs you, the section names what is in play, and with nothing in play at all it is
// absent. Reads only what the drawings above put on the page.
function drawWaiting(featured) {
  const section = document.getElementById('waiting');
  const head = document.getElementById('wait-head');
  const rows = section.querySelectorAll('.waitRow');
  const quiet = section.querySelectorAll('#between-quiet > *, #challenges-quiet > *');
  head.textContent = rows.length ? 'Waiting on you' : 'Games in play';
  const rule = document.getElementById('wait-rule');
  rule.textContent = document.querySelector('#carry .waitRow') ? IDLE_RULE : '';
  section.hidden = !rows.length && !quiet.length;
  section.classList.toggle('lone', !featured);
  if (section.hidden) return false;
  const first = section.querySelector('.waitRow .main');
  if (first) first.classList.add('primary');
  return !!rows.length;
}

async function draw() {
  let answer;
  try { answer = await read('/api/profile'); } catch (error) { say(error.message); return; }
  if (!answer.ok) { say('Sign in to see your profile.'); return; }
  const payload = answer.data;
  NAMES = {};
  for (const game of payload.games) NAMES[game.hash] = named(game);
  DELETED = new Set(payload.deleted || []);
  PIECE_ART = payload.pieceArt || {};
  const made = payload.yourGames || [];
  const columns = payload.workshop || [];
  drawBand(payload.page);
  drawCarryOn(payload.carryOn || []);
  drawBetween(payload.between);
  drawChallenges(payload.challenges);
  const waits = drawWaiting(drawFeatured(made, columns));
  const busy = made.length > 0 || !document.getElementById('waiting').hidden;
  drawStart(payload.counts, payload.games, busy, payload.today);
  // The workshop is drawn for anybody past the start: with nothing waiting, its *Start a
  // new game* is the page's one primary.
  if (busy || payload.counts.tried) drawYourGames(made, columns, !waits);
  drawMoveEmail(payload.moveEmail);
  drawTakesChallengers(payload.takesChallengers);
  drawInvite(payload.invites);
  drawYourPage(payload.page, payload.favourites);
  drawBackToDraft(payload.backToDraft || []);
  drawCounts(payload.counts);
  drawGames(payload.games, payload.counts.tried > 0);
  drawUntried(payload.games, payload.counts.tried > 0);
  drawDone(payload.milestones || [], payload.badges || []);
  drawProven(payload.proven || [], payload.provenShare);
  const record = document.getElementById('record-part');
  record.hidden = !payload.counts.tried && !(payload.proven || []).length;
  drawLately(payload.lately || []);
  // Side by side only while both halves hold something.
  document.getElementById('late-part').classList.toggle('lone',
    record.hidden || document.getElementById('lately').hidden);
}

// The band's way to your page as other members see it (R1, UP4). With the page's own
// sections drawn under `#members-view` by the server (`pages.showcase`, the very function
// `/players/<handle>` draws with), the band is a switch, *your view · as members see it*;
// without them, a link to the page's address. `#as-members` in the address opens on it.
function drawBand(address) {
  const where = document.getElementById('band-links');
  if (!where) return;
  where.replaceChildren();
  const theirs = document.getElementById('members-view');
  if (theirs && SHOWCASE) {
    const own = document.getElementById('own-view');
    const group = el('span', 'viewSwitch');
    group.setAttribute('role', 'group');
    group.setAttribute('aria-label', 'whose view');
    const mine = el('button', '', 'your view');
    const shown = el('button', '', 'as members see it');
    for (const button of [mine, shown]) button.type = 'button';
    const pick = (members) => {
      own.hidden = members;
      theirs.hidden = !members;
      mine.classList.toggle('on', !members);
      shown.classList.toggle('on', members);
      mine.setAttribute('aria-pressed', String(!members));
      shown.setAttribute('aria-pressed', String(members));
      if (members) fillStills(theirs, SHOWCASE.stills, SHOWCASE.pieceArt);
    };
    mine.addEventListener('click', () => pick(false));
    shown.addEventListener('click', () => pick(true));
    group.appendChild(mine);
    group.appendChild(shown);
    where.appendChild(group);
    pick(window.location.hash === '#as-members');
    return;
  }
  if (!address) return;
  const link = el('a', '', 'your page as members see it ›');
  link.href = address;
  where.appendChild(link);
}

// Whether to offer the maintainer's page, from the one route the application's own
// header asks (`/api/me`). The server renders this page without the role, so the link
// is the one thing here the page has to go and find out.
async function maintenance() {
  try {
    const answer = await read('/api/me');
    if (!answer.ok) return;
    const me = answer.data;
    if (!me.admin) return;
    const who = document.getElementById('who');
    const link = document.createElement('a');
    link.href = '/admin';
    link.textContent = 'Maintenance';
    who.prepend(link, ' · ');
  } catch (error) { /* offline; the header says nothing rather than something wrong */ }
}
maintenance();
document.getElementById('out').addEventListener('click', async (event) => {
  event.preventDefault();
  try { await post('/api/signout', {}); } catch (error) { say(error.message); return; }
  window.location.href = '/';
});

// UP7 (`docs/reference/user-page-cold-read.md`, finding 5): the server draws the page's
// frame held back (`#own-view.waits`), one quiet line in its place, and the frame comes in
// once, whole, when the first answer from `/api/profile` has drawn, or has failed and said
// why, so nothing above a thumb moves while the page fills.
function frameDrawn() {
  const own = document.getElementById('own-view');
  if (own && own.classList) own.classList.remove('waits');
}

draw().catch(error => say(String((error && error.message) || error))).finally(frameDrawn);
