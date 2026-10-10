const { SHOWN } = JSON.parse(document.getElementById('page-data').textContent);
// Who is reading, so their own row can be marked. Nothing else on the page depends on
// it: the table is the same table for everybody.
let YOU = null;

function cell(row, text, className) {
  const td = document.createElement('td');
  td.textContent = text;
  if (className) td.className = className;
  row.appendChild(td);
  return td;
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
function gameLink(game) {
  const link = document.createElement('a');
  link.href = GAME + game.hash;
  // S12: never twelve characters of a hash.
  link.textContent = game.name || 'an unnamed game';
  return link;
}
function plural(count, one, many) { return count + ' ' + (count === 1 ? one : many); }

// One unclaimed list, as a line: how many there are, then a few of them by name. The
// names are there so somebody can pick one, not so the library is listed twice.
function openLine(games, said, rule) {
  if (!games.length) return null;
  const item = document.createElement('li');
  bold(item, plural(games.length, 'game', 'games'));
  words(item, ' ' + said + '. ');
  const some = games.slice(0, SHOWN);
  some.forEach((game, index) => {
    if (index) words(item, ', ');
    item.appendChild(gameLink(game));
  });
  const rest = games.length - some.length;
  words(item, rest ? ' and ' + rest + ' more.' : '.');
  span(item, rule, 'rule');
  return item;
}

// The opening. Every game named is one anybody can start now, so a game taken off the
// site is on neither list. With nothing left unclaimed the heading keeps a sentence
// rather than an empty list, because two noughts would read as a scoreboard.
function drawUnclaimed(open) {
  const lead = document.getElementById('open-lead');
  const where = document.getElementById('unclaimed');
  where.replaceChildren();
  lead.textContent = 'Firsts nobody has taken yet, across the '
    + plural(open.offered, 'game', 'games') + ' you can start today. Nothing here '
    + 'expires and nobody loses one by being away.';
  const items = [
    openLine(open.unplayed, 'nobody has played yet',
      'A rated game with no finished game on it. Whoever finishes the first one is '
      + 'the first to have played it here.'),
    openLine(open.unbeaten, 'nobody has beaten the strongest computer player on',
      'A rated game whose strongest computer player no person has beaten yet. One win, '
      + 'from either side, is enough.'),
  ].filter(Boolean);
  if (!items.length) {
    lead.textContent = 'Every game you can start has been played, and on every one '
      + 'somebody has beaten the strongest computer player. There is nothing left to be '
      + 'first at.';
    return;
  }
  for (const item of items) where.appendChild(item);
}

// The table. Three counts of games, each read the way a profile reads it, and the
// reader's own row marked so they can find themselves without being ranked.
//
// With nobody on it the table itself goes and the sentence stays. A heading row over no
// rows is a list of noughts nobody is on.
//
// L3 of `docs/plan-launch.md` §3.3: below `here.listedFrom` people the server sends no
// rows at all, and the section says who is here and what puts a person on the list
// instead. `here` is absent for the plain listing.
function drawPeople(people, rated, here) {
  const where = document.getElementById('people');
  where.replaceChildren();
  const listed = !here || here.listed;
  document.getElementById('people-head').textContent =
    listed ? 'Who has played' : 'Who is here';
  document.getElementById('people-rule').hidden = !listed;
  document.getElementById('people-table').className =
    listed && people.length ? 'people' : 'people empty';
  if (!listed) {
    const lead = document.getElementById('people-lead');
    lead.replaceChildren();
    words(lead, (here.members === 1 ? '1 member is' : here.members + ' members are')
      + ' here. A person is listed on this page once they finish a rated game, beside how '
      + 'many games they have tried, won, and beaten the strongest computer player on. '
      + 'The list starts when ' + here.listedFrom + ' people have. Until then, ');
    const open = document.createElement('a');
    open.href = '#unclaimed';
    open.textContent = 'what is still unclaimed';
    lead.appendChild(open);
    words(lead, ' is above, for anybody to take.');
    return;
  }
  for (const person of people) {
    const row = document.createElement('tr');
    const first = cell(row, '');
    // Their page (R1), named by the handle; somebody who left has none.
    const name = person.page ? document.createElement('a') : document.createElement('span');
    if (person.page) name.href = person.page;
    name.textContent = person.handle;
    first.appendChild(name);
    // Their generated mark (R2), before the handle.
    const mark = window.memberMark && window.memberMark(person.mark, 20);
    if (mark) name.prepend(mark);
    if (person.user === YOU) span(first, ' you', 'mark');
    cell(row, person.tried, 'num');
    cell(row, person.won, 'num');
    cell(row, person.top, 'num');
    where.appendChild(row);
  }
  document.getElementById('people-lead').textContent = people.length
    ? 'Of ' + plural(rated, 'rated game', 'rated games') + ': how many each person has '
      + 'finished a game on, how many they have won one on, and how many they have '
      + 'beaten the strongest computer player on.'
    : 'Nobody has finished a rated game yet. The first person to finish one is '
      + 'the first name here.';
}

// Q760, *Find a member*: everybody with a page, A to Z as the server sent them, each a
// sheet that is wholly the link to their page. The field narrows the list as it is typed,
// by handle or by the name of a game they made; nothing is fetched per keystroke, since
// the list is the whole membership and small. The count says what the field matched, in
// words, and an empty match says so rather than leaving a blank.
let DIRECTORY = [];

function matches(person, asked) {
  if (!asked) return true;
  if (person.handle.toLowerCase().includes(asked)) return true;
  return (person.made || []).some(game => (game.name || '').toLowerCase().includes(asked));
}

function personSheet(person) {
  const item = document.createElement('li');
  const sheet = document.createElement('a');
  sheet.className = 'person';
  sheet.href = person.page;
  const mark = window.memberMark && window.memberMark(person.mark, 40);
  const holder = document.createElement('span');
  holder.className = 'personMark';
  if (mark) holder.appendChild(mark);
  sheet.appendChild(holder);
  const name = span(sheet, person.handle, 'personName');
  if (person.you) span(name, ' (you)', 'you');
  span(sheet, 'member since ' + person.since, 'personSince');
  const made = person.made || [];
  if (made.length) {
    const more = person.moreMade ? ' and ' + plural(person.moreMade, 'more', 'more') : '';
    span(sheet, 'made ' + made.map(game => game.name || 'an unnamed game').join(', ') + more,
      'personMade');
  }
  item.appendChild(sheet);
  return item;
}

function drawDirectory() {
  const field = document.getElementById('find');
  const asked = field.value.trim().toLowerCase();
  const shown = DIRECTORY.filter(person => matches(person, asked));
  const where = document.getElementById('directory');
  where.replaceChildren(...shown.map(personSheet));
  const found = document.getElementById('found');
  if (!asked) {
    found.textContent = plural(DIRECTORY.length, 'member', 'members') + ', A to Z.';
  } else if (!shown.length) {
    found.textContent = 'Nobody matches “' + field.value.trim() + '”. A handle is '
      + 'searched anywhere in it, so a few letters are enough.';
  } else {
    found.textContent = plural(shown.length, 'member matches', 'members match')
      + ' “' + field.value.trim() + '”.';
  }
}

function wireDirectory(people, said) {
  DIRECTORY = people;
  document.getElementById('find-said').textContent = said || '';
  const field = document.getElementById('find');
  field.addEventListener('input', drawDirectory);
  // Enter with exactly one match opens that member's page.
  field.addEventListener('keydown', event => {
    if (event.key !== 'Enter') return;
    const asked = field.value.trim().toLowerCase();
    const shown = DIRECTORY.filter(person => matches(person, asked));
    if (asked && shown.length === 1) window.location.href = shown[0].page;
  });
  drawDirectory();
}

async function draw() {
  let answer;
  try { answer = await read('/api/players'); } catch (error) { say(error.message); return; }
  if (!answer.ok) { say('Sign in to see the players.'); return; }
  const payload = answer.data;
  YOU = payload.you;
  wireDirectory(payload.directory || [], payload.directorySaid);
  drawUnclaimed(payload.unclaimed);
  drawPeople(payload.players || [], payload.rated, payload);
}
draw();
