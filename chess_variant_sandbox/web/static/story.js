// The landing's story, drawn at random on each load (LP6, `docs/plan-landing.md` §4.1). A
// classic script in the single page's <head>, after the shell's: the server writes the pool
// (`<html data-stories="...">`, the stories that still hold, the same bytes for everybody) and
// this puts the one drawn on `<html>` as `data-story` (its key) and `data-story-at` (its place
// in the pool) before the first paint, so the stylesheet shows its ending in the drawn
// headline and `landing.js` keeps its part when it clones the landing.
//
// - `?story=<key>` names it.
// - A reload or a step back or forward (Navigation Timing's `type`) keeps the tab's story,
//   kept in sessionStorage under `story`, so the scroll the browser restores is in it.
// - Any other load draws again among the *other* stories, so a visitor coming back meets a
//   new one and never the same one twice running in a tab.
// Storage can be refused (a private window, blocked site data): every read and write is
// guarded, and then each load is a fresh draw. A page with no pool (no story holds) draws
// nothing, and `landing.js` keeps the page as it is.
(function () {
  var KEY = 'story';
  var root = document.documentElement;
  var pool = (root.getAttribute('data-stories') || '').split(' ').filter(Boolean);
  var drawn = null;

  function kept() {
    try { return window.sessionStorage.getItem(KEY); } catch (refused) { return null; }
  }

  function keep(key) {
    try { window.sessionStorage.setItem(KEY, key); } catch (refused) { /* this load only */ }
  }

  function asked() {
    try { return new URLSearchParams(window.location.search).get(KEY); } catch (err) { return null; }
  }

  function navigation() {
    try {
      var entry = window.performance.getEntriesByType('navigation')[0];
      return entry ? entry.type : '';
    } catch (err) { return ''; }
  }

  function draw() {
    if (drawn !== null || !pool.length) return drawn;
    var wanted = asked();
    var last = kept();
    var type = navigation();
    if (pool.indexOf(wanted) < 0) {
      if ((type === 'reload' || type === 'back_forward') && pool.indexOf(last) >= 0) {
        wanted = last;
      } else {
        var others = pool.filter(function (key) { return key !== last; });
        if (!others.length) others = pool;
        wanted = others[Math.floor(Math.random() * others.length)];
      }
    }
    drawn = wanted;
    keep(drawn);
    root.setAttribute('data-story', drawn);
    root.setAttribute('data-story-at', String(pool.indexOf(drawn)));
    return drawn;
  }

  // Whether this load is the landing's: the bare address, or `#/about` (`core.js`'s router).
  // Another of the single page's addresses (`/game`, a game's) draws only when the landing is
  // opened from it, so a load of the games never spends the tab's story.
  function landingLoad() {
    var path = window.location.pathname;
    var hash = window.location.hash.replace(/^#\/?/, '');
    var bare = path === '/' || path === '/index.html';
    return (bare && window.location.hash.length <= 1) || hash === 'about' || hash.indexOf('about/') === 0;
  }

  // The story this document drew: `landing.js` asks it when it clones the landing.
  window.landingStory = draw;
  if (landingLoad()) draw();

  // Where the reader was in the story. The landing is cloned in after the page loads and the
  // router starts each view at its top, so the browser's own restoring finds nothing to
  // scroll; the page keeps the place itself as it is left, and on a reload or a step back to
  // the same story `landing.js` asks for it once (`landingScroll`), after it cloned the story.
  var SCROLL = 'storyScroll';
  var restored = false;
  window.addEventListener('pagehide', function () {
    var shown = document.querySelector('#view .storyPart');
    try {
      if (shown && shown.getAttribute('data-story') === drawn) {
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
    if (type !== 'reload' && type !== 'back_forward') return null;
    var held = null;
    try { held = window.sessionStorage.getItem(SCROLL); } catch (refused) { return null; }
    var parts = (held || '').split(' ');
    return parts[0] === drawn && parts.length === 2 ? Number(parts[1]) || 0 : null;
  };
})();
