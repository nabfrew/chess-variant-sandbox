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

async function draw() {
  let answer;
  try { answer = await read('/api/players'); } catch (error) { say(error.message); return; }
  if (!answer.ok) { say('Sign in to see the players.'); return; }
  const payload = answer.data;
  YOU = payload.you;
  drawUnclaimed(payload.unclaimed);
  drawPeople(payload.players || [], payload.rated, payload);
}
draw();
