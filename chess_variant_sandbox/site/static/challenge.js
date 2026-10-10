// Q571, `docs/plan-return.md` §1.4 and s8 of the mock: a challenge link, opened.
//
// The page is a constant; whose challenge it is comes from `/api/challenge/link/<token>`.
// A member is offered *Accept and play*, which starts the game and opens it. Somebody
// without an account is told whose challenge it is and that accounts are by invitation,
// and offered the way to sign in, which comes back here; the link is kept in this browser
// too, so it still works after an invitation arrives and they join (`afterwards` in
// `post.js`). It is never an invitation itself. Nothing on it counts down: a challenge
// never expires (test 2).
const { token } = JSON.parse(document.getElementById('page-data').textContent);

// What the page says about a challenge, from the server's answer. Pure, so a test can
// read every state's words without a page around it.
function linkSays(found) {
  const game = found.name || 'a game';
  const counts = found.rated
    ? "it's rated: the result moves both of your ratings on it"
    : "it's a practice game, kept and rated nowhere";
  if (found.state === 'taken') {
    return { lines: [`This challenge has been taken: ${found.players[0]} and `
      + `${found.players[1]} are playing it.`], play: found.match || null };
  }
  if (found.state === 'withdrawn') {
    return { lines: [`${found.from} took this challenge back, so it can no longer be `
      + 'accepted.'] };
  }
  if (found.state === 'paused') return { lines: [found.pausedSays] };
  if (found.state === 'yours') {
    return { lines: [`This is your challenge link to ${game}. Send it to the member you `
      + 'want to play, however you like: it works once, and it waits on your page until '
      + 'it is used or you withdraw it.'] };
  }
  if (found.visitor) {
    return {
      lines: [`This is a challenge from ${found.from} to a game of ${game}. Sign in to `
        + 'accept it.',
      `Accounts are by invitation for now; ${found.from} can ask for one for you, or you `
        + 'can request one yourself. Once you have joined, this link still works.'],
      signin: true,
    };
  }
  return {
    lines: [`${found.from} challenges you to ${game}: ${found.seat}, and ${counts}.`],
    accept: true,
    rules: found.rules || [],
  };
}

function paragraph(text, kind) {
  const line = document.createElement('p');
  if (kind) line.className = kind;
  line.textContent = text;
  return line;
}

function draw(found) {
  const where = document.getElementById('challenge');
  where.replaceChildren();
  const says = linkSays(found);
  for (const line of says.lines) where.appendChild(paragraph(line));
  if (says.rules && says.rules.length) {
    where.appendChild(paragraph('How a game between two people goes:', 'muted small'));
    const list = document.createElement('ul');
    list.className = 'muted small';
    for (const rule of says.rules) {
      const item = document.createElement('li');
      item.textContent = rule;
      list.appendChild(item);
    }
    where.appendChild(list);
  }
  const row = document.createElement('div');
  row.className = 'row';
  if (says.accept) {
    const accept = document.createElement('button');
    accept.className = 'primary';
    accept.textContent = 'Accept and play';
    accept.addEventListener('click', async () => {
      say('');
      accept.disabled = true;
      try {
        const made = await post('/api/challenge/link/' + encodeURIComponent(token) + '/accept', {});
        window.location.href = PLAY + made.id;
      } catch (error) { accept.disabled = false; say(error.message); }
    });
    row.appendChild(accept);
    const later = document.createElement('a');
    later.href = '/game';
    later.textContent = 'not now';
    row.appendChild(later);
  }
  if (says.signin) {
    try { localStorage.setItem(HELD_LINK, '/c/' + token); } catch (error) { /* kept nowhere */ }
    const signin = document.createElement('a');
    signin.href = '/signin?then=' + encodeURIComponent('/c/' + token);
    signin.textContent = 'Sign in';
    row.appendChild(signin);
    // Q831: the wall says accounts are by invitation, so the way to ask for one is here.
    const ask = document.createElement('a');
    ask.href = '/request-invitation';
    ask.textContent = 'Request an invitation';
    row.appendChild(ask);
  }
  if (says.play) {
    const open = document.createElement('a');
    open.href = PLAY + says.play;
    open.textContent = 'Open the game';
    row.appendChild(open);
  }
  const look = document.createElement('a');
  look.href = GAME + found.game;
  look.textContent = 'Look at ' + (found.name || 'the game');
  row.appendChild(look);
  where.appendChild(row);
}

// Only the request is caught: a fault in drawing is a fault, not a sentence for the reader.
async function load() {
  let answer;
  try {
    answer = await read('/api/challenge/link/' + encodeURIComponent(token));
  } catch (error) {
    document.getElementById('challenge').replaceChildren(paragraph(error.message));
    return;
  }
  if (!answer.ok) {
    const said = (answer.data && answer.data.error) || 'There is no challenge at this address.';
    document.getElementById('challenge').replaceChildren(paragraph(said));
    return;
  }
  draw(answer.data);
}

load();
