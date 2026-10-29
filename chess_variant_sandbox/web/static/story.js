// The landing's story, drawn at random each time the page is opened (LP6,
// `docs/archive/plan-landing.md` §4.1; Q1056). A classic script in the single page's <head>,
// after the shell's: the server writes the pool (`<html data-stories="...">`, the stories that
// still hold, the same bytes for everybody) and this puts the one drawn on `<html>` as
// `data-story` (its key) and `data-story-at` (its place in the pool) before the first paint, so
// the stylesheet shows its ending in the drawn headline and `landing.js` keeps its part when it
// clones the landing.
//
// - `?story=<key>` names it (a shared link, the play screen's way back to its story).
// - Every other opening, a reload included, is a fresh draw, uniform over the whole pool and
//   independent of what this tab showed before: nothing is kept between loads. (Until Q1056 a
//   reload or a step back kept the tab's story from sessionStorage, and any other load drew
//   among the others only, so a story never came twice running.)
// - The landing's switch (`landing.js`, `.storySwitch`) moves to another story in this document
//   through `landingStoryTo`; it is not kept either, so the next opening draws again.
// A page with no pool (no story holds) draws nothing, and `landing.js` keeps the page as it is.
(function () {
  var KEY = 'story';
  var root = document.documentElement;
  var pool = (root.getAttribute('data-stories') || '').split(' ').filter(Boolean);
  var drawn = null;
  var named = false;

  // A place in a pool of `n`, each equally likely: `Math.random()` is in [0, 1), so its floor
  // over `n` is 0 .. n-1 with equal widths; the `min` only guards a stream that returned 1.
  function pick(n) {
    return Math.min(n - 1, Math.floor(Math.random() * n));
  }

  function asked() {
    try { return new URLSearchParams(window.location.search).get(KEY); } catch (err) { return null; }
  }

  function show(key) {
    drawn = key;
    root.setAttribute('data-story', drawn);
    root.setAttribute('data-story-at', String(pool.indexOf(drawn)));
    return drawn;
  }

  function draw() {
    if (drawn !== null || !pool.length) return drawn;
    var wanted = asked();
    named = pool.indexOf(wanted) >= 0;
    return show(named ? wanted : pool[pick(pool.length)]);
  }

  // Whether this load is the landing's: the bare address, or `#/about` (`core.js`'s router).
  // Another of the single page's addresses (`/game`, a game's) draws only when the landing is
  // opened from it.
  function landingLoad() {
    var path = window.location.pathname;
    var hash = window.location.hash.replace(/^#\/?/, '');
    var bare = path === '/' || path === '/index.html';
    return (bare && window.location.hash.length <= 1) || hash === 'about' || hash.indexOf('about/') === 0;
  }

  // The story this document drew: `landing.js` asks it when it clones the landing.
  window.landingStory = draw;
  // The switch: show the story `step` places on from the one shown (wrapping round the pool),
  // and answer its key. A named story is no longer named once the reader moves off it.
  window.landingStoryTo = function (step) {
    if (draw() === null) return null;
    named = false;
    var at = pool.indexOf(drawn);
    return show(pool[(at + step % pool.length + pool.length) % pool.length]);
  };
  window.landingStoryPick = pick;
  if (landingLoad()) draw();

  // Where the reader was in a *named* story (`?story=`). The landing is cloned in after the page
  // loads and the router starts each view at its top, so the browser's own restoring finds
  // nothing to scroll; the page keeps the place itself as it is left, and on a reload or a step
  // back to the same address `landing.js` asks for it once (`landingScroll`). A drawn story is
  // a new draw on a reload, so it starts at the top.
  var SCROLL = 'storyScroll';
  var restored = false;

  function navigation() {
    try {
      var entry = window.performance.getEntriesByType('navigation')[0];
      return entry ? entry.type : '';
    } catch (err) { return ''; }
  }

  window.addEventListener('pagehide', function () {
    var shown = document.querySelector('#view .storyPart');
    try {
      if (named && shown && shown.getAttribute('data-story') === drawn) {
        window.sessionStorage.setItem(SCROLL, drawn + ' ' + Math.round(window.scrollY));
      } else {
        window.sessionStorage.removeItem(SCROLL);
      }
    } catch (refused) { /* the top, then */ }
  });
  window.landingScroll = function () {
    if (restored) return null;
    restored = true;
    var type = navigation();
    if (!named || (type !== 'reload' && type !== 'back_forward')) return null;
    var held = null;
    try { held = window.sessionStorage.getItem(SCROLL); } catch (refused) { return null; }
    var parts = (held || '').split(' ');
    return parts[0] === drawn && parts.length === 2 ? Number(parts[1]) || 0 : null;
  };
})();
