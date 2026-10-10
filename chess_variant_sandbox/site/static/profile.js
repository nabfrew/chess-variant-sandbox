const { LINES, SHOWN } = JSON.parse(document.getElementById('page-data').textContent);
// What each game is called, by hash. Built from the payload's own game rows, which
// cover every rated game, so *still open* can send hashes alone and every one of them
// resolves here.
let NAMES = {};

// The games whose rules were deleted. Sent as hashes rather than worked out from a name,
// because *deleted game* is a name a game could in principle be given.
let DELETED = new Set();

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
function day(stamp) { return stamp ? String(stamp).slice(0, 10) : ''; }
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
// score nobody has started (L2 of `docs/plan-launch.md` §3.2). `drawStart` says what to
// do in their place.
function drawCounts(counts) {
  const where = document.getElementById('counts');
  where.replaceChildren();
  if (!counts.tried) return;
  where.appendChild(count('games tried', counts.tried));
  where.appendChild(count('games won', counts.won));
  where.appendChild(count('strongest beaten', counts.top));
  // S2, absent at nought rather than shown as one, because nobody is behind for
  // topping no game's table.
  if (counts.highest) {
    where.appendChild(count('games where you are top rated', counts.highest));
  }
}

// L2, a profile with nothing on it: one line on what to do, and the games nobody has
// beaten yet as an offer, a few by name. Nothing here counts down or counts what the
// person has not done; once a rated game is finished the counts take its place.
const START = '/game';
function drawStart(counts, rows) {
  const where = document.getElementById('start');
  where.replaceChildren();
  if (counts.tried) return;
  const line = document.createElement('p');
  line.className = 'lead';
  words(line, 'Nothing here yet. ');
  const play = document.createElement('a');
  play.href = START;
  play.textContent = 'Play a rated game';
  line.appendChild(play);
  words(line, ' and it starts here.');
  where.appendChild(line);
  const open = rows.filter(game => game.unbeaten === true && !game.played
    && !game.retired && !DELETED.has(game.hash));
  if (!open.length) return;
  const offer = document.createElement('p');
  bold(offer, open.length + (open.length === 1 ? ' game nobody has beaten yet'
    : ' games nobody has beaten yet'));
  words(offer, ': ');
  const some = inOrder(open).slice(0, SHOWN);
  some.forEach((game, index) => {
    if (index) words(offer, ', ');
    offer.appendChild(gameLink(game.hash, named(game)));
  });
  const rest = open.length - some.length;
  words(offer, rest ? ' and ' + rest + ' more. ' : '. ');
  words(offer, 'Beat the strongest computer player on one and you are the first.');
  where.appendChild(offer);
}

// The table is hidden with the counts: before a first finished game every row of it
// would read *none yet*.
function drawGames(rows, started = true) {
  document.getElementById('record').hidden = !started;
  document.getElementById('record-under').hidden = !started;
  const games = document.getElementById('games');
  games.replaceChildren();
  for (const game of inOrder(rows)) {
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

// B2. One rated game you left running, as a line. The name is the link, and it goes to
// the board rather than to the game's page: the point of the line is the game in
// progress, and the game's page is one click on from it.
//
// An entry whose rules the store will not hand back -- a deleted game, or one taken off
// the site -- draws no link at all. There is nothing to replay the moves through, so no
// board can be drawn, and the sentence says so rather than offering a link that refuses.
function carryItem(entry) {
  const item = document.createElement('li');
  const name = entry.name || nameOf(entry.game);
  if (entry.openable) {
    const link = document.createElement('a');
    link.href = PLAY + entry.id;
    link.textContent = name;
    item.appendChild(link);
  } else {
    bold(item, name);
  }
  // S8: a named opponent always (the server's `label`), and turns, each player's move
  // being one.
  words(item, ' \u2014 against ' + (entry.label || 'the computer') + ', playing '
    + (entry.seat === 0 ? 'first' : 'second') + ', ' + entry.plies + ' turn'
    + (entry.plies === 1 ? '' : 's') + ' played'
    + (entry.moved ? ', last moved ' + day(entry.moved) : ', not moved yet') + '.');
  if (!entry.openable) {
    span(item, 'These rules are gone, so the board cannot be drawn. The game stays open '
      + 'until it is resigned after thirty days without a move.', 'rule');
  }
  return item;
}

// The whole section, newest game first, which is the order the server sends. Absent when
// nothing is running, because an empty heading would read as a nought.
function drawCarryOn(entries) {
  const where = document.getElementById('carry');
  where.replaceChildren();
  if (!entries.length) return;
  const heading = document.createElement('h2');
  heading.textContent = 'Games you left running';
  where.appendChild(heading);
  const lead = document.createElement('p');
  lead.className = 'under';
  lead.textContent = 'Rated games still open. Each one is resigned after thirty idle '
    + 'days, or when you start another rated game on the same game.';
  where.appendChild(lead);
  const list = document.createElement('ul');
  list.className = 'lines';
  for (const entry of entries) list.appendChild(carryItem(entry));
  where.appendChild(list);
}

// Q570, Q627, Q572. *Correspondence* (s6 of the mock): the line of what you finished
// (`docs/plan-return.md` §1.6), absent until there is one, then your running games
// against other people under *Your move* and *Their move*, in the server's order
// (`correspondence.started_order`: your move first, then theirs, each by the time left to
// move). When the last move landed and the time left to move are facts in the server's
// words (the dated exception to test 2): nothing here counts down, no count in a heading,
// no colour that alarms. The rule is said once, as a rule. Each part is absent when it
// has nothing in it; the section's heading stays for the email box under it.
function drawBetween(between) {
  const where = document.getElementById('between');
  if (!where) return;
  where.replaceChildren();
  const running = (between && between.running) || [];
  const finished = between && between.finished;
  if (finished) {
    const line = document.createElement('p');
    line.className = 'under';
    line.textContent = finished;
    where.appendChild(line);
  }
  if (!running.length) return;
  const part = (title, entries) => {
    if (!entries.length) return;
    const heading = document.createElement('h3');
    heading.textContent = title;
    where.appendChild(heading);
    const list = document.createElement('ul');
    list.className = 'lines';
    for (const entry of entries) {
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = PLAY + entry.id;
      link.textContent = entry.name || nameOf(entry.game);
      // The opponent's generated mark (R2), at the head of the line.
      const mark = window.memberMark && window.memberMark(entry.opponentMark, 18);
      if (mark) item.appendChild(mark);
      item.appendChild(link);
      words(item, ' against ');
      person(item, entry.opponent);
      words(item, ' \u00b7 ' + (entry.rated ? 'rated' : 'practice')
        + ', last moved ' + day(entry.moved) + '. ');
      if (entry.toMove) words(item, entry.toMove);
      list.appendChild(item);
    }
    where.appendChild(list);
  };
  part('Your move', running.filter(entry => entry.yourMove));
  part('Their move', running.filter(entry => !entry.yourMove));
  const rule = document.createElement('p');
  rule.className = 'under';
  rule.textContent = 'No clock: each of you moves when you like. ' + between.rule;
  where.appendChild(rule);
}

// Q572, `docs/plan-return.md` §1.5 and s6 of the mock. The one email's box, off unless
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
  label.style.display = 'block';
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
  const heading = document.createElement('h2');
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

// Q606, `docs/plan-return.md` §3 C4 (R10): *list me on the games I made as taking
// challengers*, one switch, off unless you switch it on, and absent for somebody with no
// published game of their own. Saved at once and said in the note, as the boxes above are.
function drawTakesChallengers(box) {
  const where = document.getElementById('takes-challengers');
  if (!where) return;
  where.replaceChildren();
  if (!box || !(box.games || []).length) return;
  const label = document.createElement('label');
  label.style.display = 'block';
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

// Q596, `docs/plan-return.md` §6.2 (R1): your page as other members see it, and its one
// box, *show my five-star games on my page*, off unless you tick it. Saved at once and
// said in the note, as the email box is.
function drawYourPage(address, box) {
  const where = document.getElementById('your-page');
  if (!where || !address) return;
  where.replaceChildren();
  const heading = document.createElement('h2');
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
  label.style.display = 'block';
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

// Q571, `docs/plan-return.md` §1.4 and s6 of the mock. *A challenge for you* with accept
// and decline, *Challenges you sent* with withdraw (a link with its address to copy
// again), and *Open to a game of* with remove. Each part is absent when it has nothing
// in it, and nothing here counts down: a challenge waits until it is answered (test 2).
// A declined challenge says so and no more.
function drawChallenges(asked) {
  const where = document.getElementById('challenges');
  if (!where) return;
  where.replaceChildren();
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
    where.appendChild(it);
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
    where.appendChild(line);
  }
  if (forYou.length) {
    heading('A challenge for you');
    const list = document.createElement('ul');
    list.className = 'lines';
    for (const entry of forYou) {
      const item = document.createElement('li');
      const mark = window.memberMark && window.memberMark(entry.fromMark, 18);
      if (mark) item.appendChild(mark);
      person(bold(item, ''), entry.from);
      words(item, entry.rematch ? ' offers a rematch at ' : ' challenged you to ');
      item.appendChild(gameLink(entry.game, entry.name));
      span(item, ' · ' + entry.seat + ' · ' + (entry.rated ? 'rated' : 'practice')
        + ' · sent ' + day(entry.made) + ' ', 'muted small');
      button(item, 'Accept', 'primary', () =>
        act('/api/challenge/' + entry.id + '/accept', {}));
      button(item, 'decline', '', () => act('/api/challenge/' + entry.id + '/decline', {}));
      list.appendChild(item);
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
    where.appendChild(list);
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
    where.appendChild(line);
  }
}

// Q503b, §3.4. *Your games*: the games this person made, grouped by level in the
// server's order, each group headed by the level's word (`site/stages.py`, the one place
// a level is decided) and each game with its next step and the way to take it, which is
// the designer. Absent when there are none, as every section here is.
const DESIGN = '/#/design/';

function drawYourGames(groups) {
  const where = document.getElementById('mine');
  where.replaceChildren();
  if (!groups.length) return;
  const heading = document.createElement('h2');
  heading.textContent = 'Your games';
  where.appendChild(heading);
  for (const group of groups) {
    const level = document.createElement('h3');
    level.textContent = group.level;
    level.dataset.level = group.level;
    where.appendChild(level);
    if (group.says) {
      const lead = document.createElement('p');
      lead.className = 'under';
      lead.textContent = group.says;
      where.appendChild(lead);
    }
    const list = document.createElement('ul');
    list.className = 'lines';
    for (const game of group.games) {
      const item = document.createElement('li');
      item.appendChild(gameLink(game.hash, game.name));
      if (game.next) words(item, ' — ' + game.next);
      // Q599: how many voters chose each fixed tag, a count and never who.
      const tagged = game.tagCounts || [];
      if (tagged.length) {
        span(item, ' Tagged: ' + tagged.map(entry => entry.words + ' · ' + entry.count)
          .join(', ') + '.', 'rule');
      }
      if (game.inDesigner) {
        words(item, ' ');
        const open = document.createElement('a');
        open.href = DESIGN + game.hash;
        open.textContent = 'open in the designer';
        item.appendChild(open);
      }
      list.appendChild(item);
    }
    where.appendChild(list);
  }
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
function milestoneItem(line) {
  const item = document.createElement('li');
  const said = LINES[line.kind] || [line.kind, 'on'];
  bold(item, said[0]);
  words(item, ', ' + said[1] + ' ');
  item.appendChild(gameLink(line.game, line.name));
  // A designer's line (D1 to D4) has no seat; D4 says who forked it.
  if (line.by) { words(item, ', by '); person(item, line.by); }
  const seat = line.seat === 0 || line.seat === 1 ? ', ' + seatWord(line.seat) : '';
  words(item, ', ' + day(line.when) + seat + '.');
  span(item, line.rule, 'rule');
  return item;
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
  for (const line of lines) items.push([line.when || '', milestoneItem(line)]);
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

// One bin of one axis. The counts first, then a few of the untried games by name,
// because the line is there so a person can pick one rather than to list the library
// again. A bin with nothing open at all draws nothing.
function openLine(kind) {
  const said = [];
  if (kind.untried.length) said.push(kind.untried.length + ' untried');
  if (kind.unwon.length) said.push(kind.unwon.length + ' not yet won');
  if (kind.unbeaten.length) said.push(kind.unbeaten.length + ' still Unbeaten');
  if (!said.length) return null;
  const item = document.createElement('li');
  // S6: the kind in the game page's own words, the server's (`kind.words`).
  bold(item, kind.words || kind.bin);
  if (!kind.tried) span(item, ' (a kind you have not tried)', 'mark');
  words(item, ' \u2014 ' + said.join(', '));
  const some = kind.untried.slice(0, SHOWN);
  if (some.length) {
    words(item, '. ');
    some.forEach((hash, index) => {
      if (index) words(item, ', ');
      item.appendChild(gameLink(hash, nameOf(hash)));
    });
    const rest = kind.untried.length - some.length;
    words(item, rest ? ' and ' + rest + ' more.' : '.');
  }
  return item;
}

// M6's *still open for you*, by grid axis. An axis whose bins are all closed is left
// out, and so is the whole section when nothing anywhere is open.
function drawStillOpen(axes) {
  const where = document.getElementById('open');
  where.replaceChildren();
  const drawn = [];
  for (const axis of axes) {
    const items = axis.kinds.map(openLine).filter(Boolean);
    // S6: headed by the question the kind answers (*How the game is won*), the server's.
    if (items.length) drawn.push([axis.asks || axis.axis, items]);
  }
  if (!drawn.length) return;
  const heading = document.createElement('h2');
  heading.textContent = 'Still open for you';
  where.appendChild(heading);
  const lead = document.createElement('p');
  lead.className = 'under';
  lead.textContent = 'Rated games you have not tried, have not yet won, and that nobody '
    + 'has beaten the strongest computer player on, grouped by kind of game. The games '
    + 'named are the untried ones. Nothing here expires; the list grows when the library '
    + 'does and shrinks as you play.';
  where.appendChild(lead);
  for (const [axis, items] of drawn) {
    const name = document.createElement('h3');
    name.textContent = axis;
    where.appendChild(name);
    const list = document.createElement('ul');
    list.className = 'open';
    for (const item of items) list.appendChild(item);
    where.appendChild(list);
  }
}

async function draw() {
  let answer;
  try { answer = await read('/api/profile'); } catch (error) { say(error.message); return; }
  if (!answer.ok) { say('Sign in to see your profile.'); return; }
  const payload = answer.data;
  NAMES = {};
  for (const game of payload.games) NAMES[game.hash] = named(game);
  DELETED = new Set(payload.deleted || []);
  drawCounts(payload.counts);
  drawStart(payload.counts, payload.games);
  drawCarryOn(payload.carryOn || []);
  drawBetween(payload.between);
  drawChallenges(payload.challenges);
  drawMoveEmail(payload.moveEmail);
  drawTakesChallengers(payload.takesChallengers);
  drawInvite(payload.invites);
  drawYourPage(payload.page, payload.favourites);
  drawYourGames(payload.yourGames || []);
  drawBackToDraft(payload.backToDraft || []);
  drawGames(payload.games, payload.counts.tried > 0);
  drawDone(payload.milestones || [], payload.badges || []);
  drawStillOpen(payload.stillOpen || []);
  if (!payload.games.length) say('You have no rated games yet. Pick one and play it.');
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
draw();
