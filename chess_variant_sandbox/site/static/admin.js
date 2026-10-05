function cell(row, text, className) {
  const td = document.createElement('td');
  td.textContent = text;
  if (className) td.className = className;
  row.appendChild(td);
  return td;
}
function button(label, grave, run) {
  const it = document.createElement('button');
  it.className = grave ? 'act grave' : 'act';
  it.textContent = label;
  it.addEventListener('click', run);
  return it;
}
function span(parent, said, className) {
  const it = document.createElement('span');
  it.textContent = said;
  if (className) it.className = className;
  parent.appendChild(it);
  return it;
}
function day(stamp) { return stamp ? stamp.slice(0, 10) : '\u2014'; }

// `note` is where the answer is said: the page's own note at the top unless the table
// pressed from has one beside it.
async function act(body, note) {
  const tell = note
    ? (text) => { document.getElementById(note).textContent = text; }
    : say;
  tell('');
  try {
    const answer = await post('/api/admin', body);
    tell(answer.said);
    await draw();
  } catch (error) { tell(error.message); }
}

function people(rows) {
  const body = document.getElementById('people');
  body.replaceChildren();
  for (const person of rows) {
    const row = document.createElement('tr');
    cell(row, person.handle);
    cell(row, person.email || '\u2014');
    cell(row, person.role);
    cell(row, person.state, 'state ' + person.state);
    cell(row, person.sessions, 'num');
    cell(row, person.invites, 'num');
    giving(cell(row, '', 'num'), person);
    cell(row, person.matches, 'num');
    const acts = cell(row, '');
    if (person.state === 'active') {
      acts.appendChild(button('kill', true, () => act({act: 'kill-user', user: person.id})));
    } else if (person.state === 'killed') {
      acts.appendChild(button('let back in', false,
        () => act({act: 'revive-user', user: person.id})));
    }
    if (person.state !== 'left') {
      acts.appendChild(button('forget', true, () => {
        const asked = 'Forget ' + person.handle + '? The address and the password hash '
          + 'are deleted and every match becomes "a former player". This cannot be undone.';
        if (window.confirm(asked)) act({act: 'forget-user', user: person.id});
      }));
    }
    body.appendChild(row);
  }
}

// Q719: how many invitations this person may send themselves, as a number set here. Only
// somebody who can sign in can send one, so the others read the number and no box.
function giving(where, person) {
  if (person.state !== 'active') { where.textContent = person.toGive; return; }
  const box = document.createElement('input');
  box.type = 'number';
  box.min = '0';
  box.max = '50';
  box.step = '1';
  box.value = String(person.toGive);
  box.style.width = '4.5em';
  box.setAttribute('aria-label', 'invitations ' + person.handle + ' can send');
  where.appendChild(box);
  where.appendChild(button('set', false, () => {
    const count = Number(box.value);
    act({act: 'set-invites', user: person.id, count: Number.isInteger(count) ? count : -1});
  }));
}

const ACCEPTED = {accepted: 'yes', open: 'not yet', expired: 'no, expired'};

function invites(rows) {
  const body = document.getElementById('invites');
  body.replaceChildren();
  for (const invite of rows) {
    const row = document.createElement('tr');
    cell(row, invite.email || '\u2014');
    cell(row, day(invite.created));
    cell(row, invite.issuer || '\u2014');
    cell(row, day(invite.expiry));
    cell(row, ACCEPTED[invite.state] || invite.state, 'state ' + invite.state);
    body.appendChild(row);
  }
}

function inviteSay(text, done) {
  const note = document.getElementById('invite-note');
  note.textContent = text;
  note.className = done ? 'note done' : 'note';
}

// The button stays disabled until the answer is back, so a double click sends one mail.
document.getElementById('invite-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const field = document.getElementById('invite-email');
  const sending = document.getElementById('invite-send');
  inviteSay('');
  sending.disabled = true;
  try {
    const answer = await post('/api/admin', {act: 'invite', email: field.value});
    field.value = '';
    inviteSay(answer.said, true);
    await draw();
  } catch (error) {
    inviteSay(error.message);
  } finally {
    sending.disabled = false;
  }
});

function games(rows) {
  const body = document.getElementById('games');
  body.replaceChildren();
  for (const game of rows) {
    const row = document.createElement('tr');
    const first = cell(row, '');
    const link = document.createElement('a');
    link.href = GAME + game.hash;
    link.textContent = game.name || game.hash.slice(0, 12);
    first.appendChild(link);
    const mark = document.createElement('code');
    mark.className = 'hash';
    mark.textContent = ' ' + game.hash.slice(0, 8);
    first.appendChild(mark);
    // A draft names who saved it, and says so when it is off the listing.
    const made = game.origin === 'designed' && game.owner ? 'by ' + game.owner : '';
    const hidden = game.visibility === 'public' ? '' : game.visibility;
    cell(row, [game.origin, made, hidden].filter(Boolean).join(', '));
    cell(row, game.matches, 'num');
    cell(row, game.state, 'state ' + game.state);
    const acts = cell(row, '');
    // The buttons act on the kill switch, not on the state word: a retired game can
    // still be killed, and a killed one is put back either way. A deleted game has no
    // rules left, so neither button has anything to act on and the row carries none.
    if (game.gone) {
      span(acts, 'its rules are gone', 'muted');
    } else if (!game.killed) {
      acts.appendChild(button('take off', true,
        () => act({act: 'kill-game', hash: game.hash})));
    } else {
      acts.appendChild(button('put back', false,
        () => act({act: 'revive-game', hash: game.hash})));
    }
    body.appendChild(row);
  }
}

// Q649. Every score a new measurement replaced: one row a replacement, newest game first.
// Each line is a past value with its basis, never a score; every cell is a text node.
function scoreHistory(rows) {
  const body = document.getElementById('score-history');
  body.replaceChildren();
  const empty = document.getElementById('score-history-empty');
  empty.textContent = rows.length ? '' : 'No score has been replaced yet.';
  for (const game of rows) {
    for (const was of game.entries || []) {
      const row = document.createElement('tr');
      const first = cell(row, '');
      const link = document.createElement('a');
      link.href = GAME + game.hash;
      link.textContent = game.name || game.hash.slice(0, 12);
      first.appendChild(link);
      const number = was.index === null || was.index === undefined
        ? 'no score' : Number(was.index).toFixed(4);
      cell(row, number + ' (' + was.runs + ' run' + (was.runs === 1 ? '' : 's') + ', '
        + (was.weights_version || '?') + ')', 'num');
      cell(row, day(was.measured));
      cell(row, day(was.replaced));
      cell(row, was.players + '; moved: ' + (was.moved || []).join(', '));
      cell(row, (was.reasons || []).join('; '));
      cell(row, was.convergence || '—');
      body.appendChild(row);
    }
  }
}

// EV6 (Q690). How much each stored score has moved between its revisions at one weights
// version, beside its interval, as `history.volatility` reads and words it. No cut-off and
// no colour: the page compares nothing. A game with fewer than two scores reads a dash.
function scoreVolatility(rows) {
  const body = document.getElementById('score-volatility');
  body.replaceChildren();
  document.getElementById('score-volatility-empty').textContent =
    rows.length ? '' : 'No game has a stored score here.';
  const figure = value => value === null || value === undefined
    ? '\u2014' : Number(value).toFixed(3);
  for (const game of rows) {
    const row = document.createElement('tr');
    const first = cell(row, '');
    const link = document.createElement('a');
    link.href = GAME + game.hash;
    link.textContent = game.name || game.hash.slice(0, 12);
    first.appendChild(link);
    cell(row, String(game.revisions) + (game.weights_version
      ? ' at ' + game.weights_version : ''), 'num');
    cell(row, figure(game.spread), 'num');
    cell(row, figure(game.largest_step), 'num');
    cell(row, game.interval_half === null || game.interval_half === undefined
      ? '\u2014' : '\u00b1' + figure(game.interval_half), 'num');
    cell(row, game.said);
    body.appendChild(row);
  }
}

// Q705. The refresher's status, worded by the server (`web/refreshing.py`): the lines in
// order, then each game it refused and will not try again, newest first. No status file
// is a refresher that is not running, said in a line like any other.
function refresherStatus(status) {
  const said = document.getElementById('refresher-said');
  said.replaceChildren();
  for (const line of (status && status.said) || ['The refresher is not running here.']) {
    const item = document.createElement('li');
    item.textContent = line;
    said.appendChild(item);
  }
  const body = document.getElementById('refresher-refused');
  body.replaceChildren();
  const refused = (status && status.refused) || [];
  document.getElementById('refresher-refused-empty').textContent =
    refused.length ? '' : 'It has refused no game.';
  for (const one of refused) {
    const row = document.createElement('tr');
    const first = cell(row, '');
    const link = document.createElement('a');
    link.href = GAME + one.hash;
    link.textContent = one.name || one.hash.slice(0, 12);
    first.appendChild(link);
    cell(row, one.needs || '\u2014');
    cell(row, one.why);
    cell(row, one.reason || '\u2014');
    cell(row, day(one.at));
    body.appendChild(row);
  }
}

// Q503d. Every report, newest first. The button is the games table's own kill switch,
// read off the same two flags, so a report and the games table can never disagree about
// what the press does. The note is a text node like every other cell.
function reports(rows) {
  const body = document.getElementById('reports');
  body.replaceChildren();
  for (const one of rows) {
    const row = document.createElement('tr');
    const first = cell(row, '');
    const link = document.createElement('a');
    link.href = GAME + one.hash;
    link.textContent = one.name || one.hash.slice(0, 12);
    first.appendChild(link);
    cell(row, one.label);
    cell(row, one.note || '\u2014');
    cell(row, one.handle);
    cell(row, day(one.created));
    const acts = cell(row, '');
    if (one.gone) {
      span(acts, 'its rules are gone', 'muted');
    } else if (!one.killed) {
      acts.appendChild(button('take off', true,
        () => act({act: 'kill-game', hash: one.hash}, 'reports-note')));
    } else {
      acts.appendChild(button('put back', false,
        () => act({act: 'revive-game', hash: one.hash}, 'reports-note')));
    }
    // Q503o. Dismissing keeps the row, so the same person still cannot report again.
    acts.appendChild(button('dismiss', false,
      () => act({act: 'dismiss-report', reporter: one.reporter, hash: one.hash},
        'reports-note')));
    body.appendChild(row);
  }
  document.getElementById('reports-empty').textContent = rows.length
    ? '' : 'Nobody has reported a game.';
}

// Q518. The reports a maintainer dismissed, most recently dismissed first, read only.
// Folded away under the open list; the summary carries the count so the fold says whether
// there is anything in it. Every cell a text node, as in the open list.
function dismissedReports(rows, count) {
  const body = document.getElementById('dismissed');
  body.replaceChildren();
  for (const one of rows) {
    const row = document.createElement('tr');
    const first = cell(row, '');
    const link = document.createElement('a');
    link.href = GAME + one.hash;
    link.textContent = one.name || one.hash.slice(0, 12);
    first.appendChild(link);
    if (one.gone) span(first, ' (its rules are gone)', 'muted');
    else if (one.killed) span(first, ' (off the site)', 'muted');
    cell(row, one.label);
    cell(row, one.note || '\u2014');
    cell(row, one.handle);
    cell(row, day(one.created));
    cell(row, day(one.dismissed));
    body.appendChild(row);
  }
  document.getElementById('dismissed-summary').textContent =
    'Dismissed reports (' + count + ')';
  document.getElementById('dismissed-said').textContent = !count
    ? 'No report has been dismissed.'
    : rows.length < count ? 'The newest ' + rows.length + ' of ' + count + '.' : '';
}

// Q730. The accounts a spot check flagged (Q503e), newest flag first. Clearing is
// reversible in effect (the next mismatch flags again) but lets the account publish, so
// it asks first and names the account. Every cell a text node.
function flagged(rows) {
  const body = document.getElementById('flagged');
  body.replaceChildren();
  for (const one of rows) {
    const row = document.createElement('tr');
    const who = one.handle || ('account ' + one.user);
    cell(row, who);
    cell(row, day(one.flagged));
    cell(row, one.found || '—', 'found');
    const acts = cell(row, '');
    acts.appendChild(button('clear the flag', false, () => {
      if (window.confirm('Let ' + who + ' publish again?')) {
        act({act: 'clear-flag', user: one.user}, 'flagged-note');
      }
    }));
    body.appendChild(row);
  }
  document.getElementById('flagged-empty').textContent = rows.length
    ? '' : 'No account is flagged.';
}

// The suggestion box, open rows first. Every cell is a text node, the suggestion itself
// included, so a suggestion made of tags is read rather than run.
//
// *Done* is reversible and gets a plain button. Deleting takes the text away for good and
// tells nobody, so it asks first and quotes what is about to go.
function suggestions(rows) {
  const body = document.getElementById('suggestions');
  body.replaceChildren();
  for (const one of rows) {
    const row = document.createElement('tr');
    cell(row, one.label);
    cell(row, one.handle);
    cell(row, day(one.created));
    cell(row, one.body);
    cell(row, one.state, 'state ' + one.state);
    const acts = cell(row, '');
    if (one.state === 'open') {
      acts.appendChild(button('mark done', false,
        () => act({act: 'done-suggestion', suggestion: one.id})));
    } else {
      acts.appendChild(button('reopen', false,
        () => act({act: 'open-suggestion', suggestion: one.id})));
    }
    acts.appendChild(button('delete', true, () => {
      const asked = 'Delete this suggestion? The text goes for good and '
        + one.handle + ' is not told.\n\n' + one.body;
      if (window.confirm(asked)) act({act: 'drop-suggestion', suggestion: one.id});
    }));
    body.appendChild(row);
  }
  document.getElementById('suggestions-empty').textContent = rows.length
    ? '' : 'Nobody has suggested anything yet.';
}

// Q503h. The latest round. An entrant the members passed carries a reason box and the
// two buttons the server allows it (`mayConfirm`, `mayVeto`); a confirmed one shows the
// `archive` command to run. Every cell is a text node, the reason included.
function round(reading) {
  const body = document.getElementById('round-entrants');
  body.replaceChildren();
  document.getElementById('round-said').textContent = reading
    ? (reading.round ? 'The round of ' + reading.words + '. ' : '') + reading.says : '';
  const heads = [...document.querySelectorAll('#round-table thead th')]
    .map(th => th.textContent);
  for (const one of (reading && reading.entrants) || []) {
    const row = document.createElement('tr');
    const first = cell(row, '');
    const link = document.createElement('a');
    link.href = GAME + one.hash;
    link.textContent = one.name;
    first.appendChild(link);
    span(first, ' by ' + one.designer, 'muted');
    if (one.carried) span(first, ' (carried from ' + one.carried + ')', 'muted');
    cell(row, one.score === null ? '—' : one.score.toFixed(3), 'num');
    cell(row, one.premise);
    cell(row, one.cost);
    cell(row, one.yes === null ? 'hidden until the close'
      : one.yes + ' yes, ' + one.no + ' no' + (one.passed ? ', passed' : ''));
    const decided = cell(row, one.decisionSays || '');
    if (one.command) {
      decided.appendChild(document.createElement('br'));
      decided.appendChild(archiveCommand(one.command));
    }
    const acts = cell(row, '', 'acts');
    if (one.mayConfirm || one.mayVeto) {
      const reason = document.createElement('input');
      reason.type = 'text';
      reason.placeholder = 'the reason, shown on its page';
      reason.maxLength = 500;
      acts.appendChild(reason);
      const decide = (act) => () => {
        if (!reason.value.trim()) { roundSay('A decision needs its reason.'); return; }
        roundAct({act, hash: one.hash, reason: reason.value});
      };
      if (one.mayConfirm) acts.appendChild(button('confirm', false, decide('confirm-entrant')));
      if (one.mayVeto) acts.appendChild(button('veto', true, decide('veto-entrant')));
    }
    if (one.why) span(acts, one.why, 'muted');
    labelled(row, heads);
    body.appendChild(row);
  }
}

// Each cell carries its column's heading, which the narrow layout prints before it; the
// first (the game) and the last (the buttons) need none.
function labelled(row, heads) {
  [...row.children].forEach((td, index) => {
    const head = heads[index] || '';
    if (head && index > 0) td.dataset.label = head;
  });
}

// The `archive` command, every word a text node. A flag is kept whole, so a narrow cell
// wraps between words and never after the `--` of `--credit`.
function archiveCommand(text) {
  const code = document.createElement('code');
  code.className = 'command';
  text.split(' ').forEach((word, index) => {
    if (index) code.appendChild(document.createTextNode(' '));
    if (word.startsWith('-')) span(code, word, 'flag');
    else code.appendChild(document.createTextNode(word));
  });
  return code;
}

function roundSay(text) { document.getElementById('round-note').textContent = text; }

async function roundAct(body) {
  roundSay('');
  try {
    const answer = await post('/api/admin', body);
    roundSay(answer.said);
    await draw();
  } catch (error) { roundSay(error.message); }
}

function limits(rows) {
  const body = document.getElementById('limits');
  body.replaceChildren();
  for (const limit of rows) {
    const row = document.createElement('tr');
    cell(row, limit.label);
    cell(row, limit.value, 'num');
    body.appendChild(row);
  }
}

function lines(id, said) {
  const list = document.getElementById(id);
  list.replaceChildren();
  for (const line of said) {
    const item = document.createElement('li');
    item.textContent = line;
    list.appendChild(item);
  }
}

function uses(id, reading) {
  document.getElementById(id + '-said').textContent = reading.said.join(' ');
  const body = document.getElementById(id);
  body.replaceChildren();
  for (const use of reading.routes) {
    const row = document.createElement('tr');
    cell(row, (use.route + ' ' + use.variant).trim());
    cell(row, use.requests, 'num');
    cell(row, use.refused, 'num');
    cell(row, use.byVisitors, 'num');
    cell(row, use.cpu.toFixed(1), 'num');
    cell(row, use.wallMean.toFixed(3), 'num');
    cell(row, use.wallMax.toFixed(3), 'num');
    cell(row, use.peopleLargest, 'num');
    cell(row, use.partialHours
      ? use.peopleHours + ' / ' + use.partialPeople + ' in ' + use.partialHours + ' partial'
      : use.peopleHours, 'num');
    body.appendChild(row);
  }
}

// Q721. A row a UTC day, newest first: counts only, every cell a number.
const DAY_COLUMNS = ['requests', 'byVisitors', 'errors', 'practice', 'withMembers',
  'players', 'joined', 'invited', 'mailRefused', 'moveEmails', 'moveEmailsFailed'];

function days(reading) {
  const body = document.getElementById('days');
  body.replaceChildren();
  for (const day of reading.rows) {
    const row = document.createElement('tr');
    cell(row, day.day);
    for (const column of DAY_COLUMNS) cell(row, day[column], 'num');
    body.appendChild(row);
  }
  lines('days-said', reading.said);
}

function usage(reading, byDay) {
  document.getElementById('use-kept').textContent = reading.kept;
  if (byDay) days(byDay);
  uses('use-day', reading.day);
  uses('use-month', reading.month);
  lines('use-show-me', reading.month.showMe.said);
  lines('use-box', reading.box.said);
  lines('use-cannot', (byDay ? byDay.cannot : []).concat(
    reading.month.cannot, reading.box.cannot));
}

// Q570, Q572. The pause on new games between people, how many are running, and the
// move emails against their daily cap. The button
// says what pressing it does, never the state alone.
function between(reading) {
  if (!reading) return;
  document.getElementById('between-said').textContent = (reading.paused
    ? 'Paused. A member reads: "' + reading.pausedSays + '" '
    : 'Open. ') + reading.running + ' game(s) between people running. '
    // Q572. The move emails of the last rolling day against the site-wide cap.
    + (reading.mailed === undefined ? '' : reading.mailed + ' of ' + reading.mailCap
      + ' move emails sent in the last day; past the cap the page is the only signal.');
  const flip = document.getElementById('between-switch');
  flip.textContent = reading.paused
    ? 'let new games between people start' : 'pause new games between people';
  flip.onclick = () => act({act: reading.paused
    ? 'resume-correspondence' : 'pause-correspondence'}, 'between-note');
}

document.getElementById('pair-form').addEventListener('submit', (event) => {
  event.preventDefault();
  act({
    act: 'pair',
    hash: document.getElementById('pair-hash').value.trim(),
    seat0: document.getElementById('pair-seat0').value.trim(),
    seat1: document.getElementById('pair-seat1').value.trim(),
  }, 'between-note');
});

async function draw() {
  let answer;
  try { answer = await read('/api/admin'); } catch (error) { say(error.message); return; }
  if (!answer.ok) { say('This page is the maintainer\'s.'); return; }
  const payload = answer.data;
  people(payload.users);
  invites(payload.invites);
  games(payload.games);
  scoreHistory(payload.history || []);
  scoreVolatility(payload.volatility || []);
  refresherStatus(payload.refresher);
  reports(payload.reports || []);
  dismissedReports(payload.dismissedReports || [], payload.dismissedCount || 0);
  flagged(payload.flagged || []);
  round(payload.round);
  between(payload.correspondence);
  suggestions(payload.suggestions || []);
  limits(payload.limits);
  usage(payload.usage, payload.days);
}

document.getElementById('revoke-all').addEventListener('click', () => {
  if (window.confirm('Sign every person out, on every device?')) {
    act({act: 'revoke-sessions'});
  }
});
draw();
