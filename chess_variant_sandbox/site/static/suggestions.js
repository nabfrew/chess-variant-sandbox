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
function day(stamp) { return stamp ? String(stamp).slice(0, 10) : ''; }

// One suggestion of your own. The text is a text node and nothing else on this page
// writes markup, so a tag in it is a tag on the screen.
function item(one) {
  const line = document.createElement('li');
  bold(line, one.label);
  words(line, ', ' + day(one.created) + '. ');
  span(line, one.body);
  if (one.state === 'done') span(line, ' Marked done ' + day(one.done) + '.', 'rule');
  return line;
}

function drawMine(rows) {
  const list = document.getElementById('mine');
  list.replaceChildren();
  for (const one of rows) list.appendChild(item(one));
  document.getElementById('empty').textContent = rows.length
    ? '' : 'You have not suggested anything yet.';
}

function drawKinds(kinds) {
  const picker = document.getElementById('kind');
  if (picker.options.length) return;
  for (const kind of kinds) {
    const option = document.createElement('option');
    option.value = kind.kind;
    option.textContent = kind.label;
    picker.appendChild(option);
  }
}

function drawLimits(limits) {
  document.getElementById('limits').textContent = 'Up to ' + limits.length
    + ' characters. You can post ' + limits.aDay + ' a day; ' + limits.left
    + ' left today.';
}

async function draw() {
  let answer;
  try { answer = await read('/api/suggestions'); } catch (error) { say(error.message); return; }
  if (!answer.ok) { say('Sign in to post a suggestion.'); return; }
  const payload = answer.data;
  drawKinds(payload.kinds);
  drawLimits(payload.limits);
  drawMine(payload.suggestions);
}

// The button stays disabled until the answer is back, so a double click posts once.
document.getElementById('go').addEventListener('click', async () => {
  const sending = document.getElementById('go');
  const text = document.getElementById('body');
  say('');
  sending.disabled = true;
  try {
    await post('/api/suggest', {
      kind: document.getElementById('kind').value,
      body: text.value,
    });
    text.value = '';
    say('Posted. The person who runs the site reads it.');
    document.getElementById('note').className = 'note done';
    await draw();
  } catch (error) {
    document.getElementById('note').className = 'note';
    say(error.message);
  } finally {
    sending.disabled = false;
  }
});
draw();
