// *What's new*: `docs/changelog.md`, day by day, newest first.
//
// One call to `GET /api/changelog` and one pass over what comes back. The payload is
// already blocks and inline runs (`web/changelog.py`), so this module decides only what an
// entry looks like -- nothing here reads markdown, and a construct the parser does not
// support arrives as text rather than as something to interpret here.
//
// **Every string from the file is set as `textContent`, never as markup.** That is the whole
// of the page's safety: an entry holding `<script>` puts those characters on screen, because
// the only way text reaches the document is through a text node. It is also why this view
// builds elements where the rest of the page writes template strings -- `esc` is a function
// somebody can forget to call, and a text node is not.

// ----------------------------------------------------------------- what's new

async function showWhatsNew() {
  stopPolling();
  SESSION = null;
  SHOWN = null;
  view.innerHTML = `
    <div class="news">
      <h2>What's new</h2>
      <div class="muted small">What this project can do now that it could not before,
        newest first. It is written as the work lands, so it is a developer's diary of a
        research project rather than release notes.</div>
      <div id="newsBody"></div>
    </div>`;
  pending('Reading the changelog…');
  const data = await api('/api/changelog');
  clearPending();
  document.getElementById('tagline').textContent = newsSummary(data);
  paintNews(document.getElementById('newsBody'), data);
}

function newsSummary(data) {
  const entries = data.days.reduce((total, day) => total + day.entries.length, 0);
  if (!entries) return 'what has changed here';
  return `${entries} ${entries === 1 ? 'entry' : 'entries'} over `
    + `${data.days.length} ${data.days.length === 1 ? 'day' : 'days'}`;
}

// The file as elements. A missing file is a sentence and not an empty page, because an
// empty page reads as *nothing has happened here* rather than as *this server has no
// copy*.
function paintNews(where, data) {
  where.textContent = '';
  if (data.note) {
    const said = document.createElement('p');
    said.className = 'muted';
    said.textContent = data.note;
    where.append(said);
    return;
  }
  data.days.forEach(day => where.append(newsDay(day)));
}

function newsDay(day) {
  const section = document.createElement('section');
  section.className = 'newsDay';
  const when = document.createElement('h3');
  when.textContent = day.words;
  // The day's entries on one sheet, the date beside it (V7).
  const sheet = document.createElement('div');
  sheet.className = 'panel newsSheet';
  day.entries.forEach(entry => sheet.append(newsEntry(entry)));
  section.append(when, sheet);
  return section;
}

function newsEntry(entry) {
  const article = document.createElement('article');
  article.className = 'newsEntry';
  if (entry.heading) {
    const heading = document.createElement('h4');
    heading.textContent = entry.heading;
    article.append(heading);
  }
  entry.blocks.forEach(block => {
    if (block.kind === 'rule') { article.append(document.createElement('hr')); return; }
    const paragraph = document.createElement('p');
    (block.runs || []).forEach(run => paragraph.append(newsRun(run)));
    article.append(paragraph);
  });
  return article;
}

// One piece of a paragraph. `em` and `code` are the two the file uses; anything else is
// its own text, which is what makes an unknown kind a sentence a reader can still read.
function newsRun(run) {
  if (run.t === 'code' || run.t === 'em') {
    const element = document.createElement(run.t === 'code' ? 'code' : 'em');
    element.textContent = run.s;
    return element;
  }
  return document.createTextNode(run.s);
}
