// The shell's own script, on every page of both servers (web/page/shell.py). A classic
// script in the <head>, before the stylesheet, so the first lines run before anything is
// painted: the viewer's chosen theme is on <html> by then and the page never flashes the
// other one (plan-visual.md §3.6, §5.1). The rest waits for the document: the footer's
// theme switch, and the phone's menu of sections.
//
// The choice is a per-viewer convenience kept in this browser's localStorage under
// `theme`: "light" or "dark", or nothing for the system's. Storage can be refused (a
// private window, blocked site data), so every read and write is guarded and the page
// simply follows the system then.
(function () {
  var KEY = 'theme';
  var root = document.documentElement;

  function stored() {
    try {
      var pick = window.localStorage.getItem(KEY);
      return pick === 'light' || pick === 'dark' ? pick : 'system';
    } catch (refused) {
      return 'system';
    }
  }

  function apply(pick) {
    if (pick === 'light' || pick === 'dark') root.setAttribute('data-theme', pick);
    else root.removeAttribute('data-theme');
  }

  apply(stored());
  // Says the script ran, so the stylesheet may fold the sections behind the menu button:
  // without it they wrap under the name instead and stay reachable.
  root.classList.add('js');

  function keep(pick) {
    try {
      if (pick === 'system') window.localStorage.removeItem(KEY);
      else window.localStorage.setItem(KEY, pick);
    } catch (refused) { /* this page only, then */ }
  }

  function wireThemes() {
    var buttons = document.querySelectorAll('[data-theme-pick]');
    function paint(pick) {
      buttons.forEach(function (button) {
        button.setAttribute('aria-pressed', String(button.dataset.themePick === pick));
      });
    }
    paint(stored());
    buttons.forEach(function (button) {
      button.addEventListener('click', function () {
        var pick = button.dataset.themePick;
        keep(pick);
        apply(pick);
        paint(pick);
        // A drawing that read its colours once (the map's canvas) repaints on this.
        document.dispatchEvent(new Event('themechange'));
      });
    });
  }

  function wireMenu() {
    var button = document.getElementById('menu');
    var nav = document.getElementById('sections');
    if (!button || !nav) return;
    button.hidden = false;
    function open(on) {
      root.classList.toggle('menu-open', on);
      button.setAttribute('aria-expanded', String(on));
    }
    button.addEventListener('click', function (event) {
      event.stopPropagation();
      open(!root.classList.contains('menu-open'));
    });
    nav.addEventListener('click', function () { open(false); });
    document.addEventListener('click', function (event) {
      if (!nav.contains(event.target)) open(false);
    });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') open(false);
    });
    // Which section the page is in, for the header's underline, from the route's first
    // part: `design` is Build, `whats-new` is What's new, the round and the landing page
    // (`about`, and the bare `/`) are none, and every other route on the application's
    // page, `/` or `/game` (Q723), is a game or the games. A site page, at another
    // address, is in none of them.
    function here() {
      var path = window.location.pathname.replace(/\/+$/, '') || '/';
      // A game's own address (G3) is the application's page, in the games.
      var game = /^\/game\/[0-9a-f]{8,64}$/.test(path);
      var page = path === '/' || path === '/index.html' || path === '/game' || game;
      var hash = window.location.hash.replace(/^#/, '');
      var part = page ? (hash.replace(/^\/?/, '').split(/[/?]/)[0] || '') : null;
      var bare = part === '' && path !== '/game' && !game && hash !== '/';
      // The people page and a member's page (Q760) are in People.
      var people = path === '/players' || path.indexOf('/players/') === 0;
      var section = people ? 'players'
        : part === null || bare || part === 'round' || part === 'about' ? null
        : part === 'design' ? '#/design' : part === 'whats-new' ? '#/whats-new' : 'game';
      nav.querySelectorAll('a').forEach(function (link) {
        var where = link.getAttribute('href').replace(/^\//, '');
        if (where === section) link.setAttribute('aria-current', 'page');
        else link.removeAttribute('aria-current');
      });
    }
    here();
    window.addEventListener('hashchange', here);
    window.addEventListener('popstate', here);
  }

  document.addEventListener('DOMContentLoaded', function () {
    wireThemes();
    wireMenu();
  });
})();

// A member's mark (R2 of archive/plan-return.md §6.3): their own invader, which the server
// generates from their account (`web/page/marks.py`) and every payload naming a person
// carries as `{d, hue}`. One drawing for both servers' pages: an <svg> element in the
// hue's token, so it follows the theme, or null for a payload without one. Built as
// elements, never markup, so a path can only ever be a path.
window.memberMark = function (mark, size) {
  if (!mark || typeof mark.d !== 'string' || !/^[MhvzZ0-9 -]*$/.test(mark.d)) return null;
  var NS = 'http://www.w3.org/2000/svg';
  var svg = document.createElementNS(NS, 'svg');
  var side = String(size || 20);
  svg.setAttribute('class', 'memberMark');
  svg.setAttribute('viewBox', '0 0 16 16');
  svg.setAttribute('width', side);
  svg.setAttribute('height', side);
  svg.setAttribute('shape-rendering', 'crispEdges');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  var path = document.createElementNS(NS, 'path');
  path.setAttribute('fill', 'var(--mk-' + (Number(mark.hue) || 1) + ')');
  path.setAttribute('d', mark.d);
  svg.appendChild(path);
  return svg;
};
