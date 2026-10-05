// The prelude every site page loads first: ask the server, and put the error where the
// person is looking rather than in a dialogue they have to dismiss.
//
// **Every request a site page makes goes through `askOnce`** (Q670, as Q659 did for the
// application's page in `web/static/page/core.js`), because the answer is not always
// JSON. While the site is redeployed or restarted, the proxy in front of it answers with a
// web page (a 502 or 503), and `response.json()` on that throws the browser's own
// sentence: *The string did not match the expected pattern* in Safari, *Unexpected token
// <* in Chrome, *Load failed* or *Failed to fetch* for a dropped connection. Such an answer
// becomes one sentence in the site's voice, the error carries `unavailable`, and the
// server's own sentence still wins whenever it sent one.
//
// **Only a GET is asked twice** (`read`), once, after a pause: it changes nothing. A POST
// (`post`) is never repeated. Some are harmless twice (signing out, the move-email switch,
// taking a game off your list each set a state), but most are not: a suggestion, an
// invitation or an accepted challenge would be made twice, and a sign-in counts against
// its limit. A person can press the button again; the page does not guess for them.
let ASK_RETRY_PAUSE = 1500;   // ms before the one repeat of a GET
const UNANSWERED = 'The site is updating or busy and did not answer. Try again in a moment.';

// One request, never thrown: what came back, or `unavailable` when nothing this page can
// read did. A refusal the server wrote (any status, with a JSON body) is an answer; an
// error page with no JSON in it is not, whatever its status.
async function askOnce(path, options) {
  let response;
  try { response = await fetch(path, options); } catch (error) { return { unavailable: true }; }
  let data;
  try { data = await response.json(); } catch (error) { return { unavailable: true }; }
  if (!response.ok && !(data && data.error) && [502, 503, 504].includes(response.status)) {
    return { unavailable: true };
  }
  return { ok: response.ok, status: response.status, data };
}

function unanswered() {
  const error = new Error(UNANSWERED);
  error.unavailable = true;
  return error;
}

// A GET. What came back, `ok` or not (a page says *Sign in* for a refusal itself), or a
// throw in the site's words when nothing readable did, twice.
async function read(path) {
  let answer = await askOnce(path, {});
  if (answer.unavailable) {
    await new Promise(done => setTimeout(done, ASK_RETRY_PAUSE));
    answer = await askOnce(path, {});
  }
  if (answer.unavailable) throw unanswered();
  return answer;
}

// A POST of JSON, asked once. The custom header is §6's answer to a cross-site POST,
// which the server refuses without it.
async function post(path, body) {
  const answer = await askOnce(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-Requested-With': 'chess-variant-sandbox'},
    body: JSON.stringify(body),
  });
  if (answer.unavailable) throw unanswered();
  if (!answer.ok) {
    throw new Error((answer.data && answer.data.error) || 'That did not work. Try again in a moment.');
  }
  return answer.data;
}
function say(text) { document.getElementById('note').textContent = text; }

// Q571. Where signing in or joining goes afterwards: the challenge link somebody opened
// before they were signed in (`pages._then` hands it over in the page's data, or the link's
// page kept it in this browser for an invitation that arrives later), else the games,
// `/game` (Q723). Only a challenge link's own shape is ever followed, so this is never a way off
// the site.
const LINK_SHAPE = /^\/c\/[A-Za-z0-9_-]{1,64}$/;
const HELD_LINK = 'challenge-link';
function afterwards() {
  let held = null;
  try {
    const data = document.getElementById('page-data');
    held = data ? JSON.parse(data.textContent).then : null;
  } catch (error) { held = null; }
  try {
    if (!held) held = localStorage.getItem(HELD_LINK);
    localStorage.removeItem(HELD_LINK);
  } catch (error) { /* storage refused: the front page */ }
  return held && LINK_SHAPE.test(held) ? held : '/game';
}
