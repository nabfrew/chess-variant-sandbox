// The page's last script: it starts the page, once every view before it has loaded.

// The name in the header goes home: the landing page, `/`, for everybody (Q718, Q723). The
// games are the header's *Games*, at `/game`.
document.getElementById('home').onclick = () => goPath('/');

// The first draw waits for `/api/me`, because what a view offers depends on the server.
identify().then(() => {
  paintIdentity();
  window.addEventListener('hashchange', onHashChange);
  window.addEventListener('popstate', onHistory);
  render();
});
