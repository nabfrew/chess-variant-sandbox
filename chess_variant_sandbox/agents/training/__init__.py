"""The learned evaluator's trainer: self-play data, a numpy trainer, quantization, stores (Q502f).

Design record: ``docs/reference/learned-evaluator.md`` (the net, the store format, and
§8 here, the trainer). ``agents/nets.py`` holds the player name, the store reader and the
integer oracle; this package writes what that module reads.

**Four steps, each its own module.**

1. ``data``: the core plays the games (``tournament.runner``, rule 2) and reports each
   position's active inputs (``Core.net_line``) -- Python replays the moves only to know
   whose turn each position is. The optional search label is the core's analysis probe
   (``Engine.root_scores``). Python answers no engine question.
2. ``model``: a float net shaped like the integer one, trained by Adam on the positions,
   in numpy, single-threaded, from a fixed seed.
3. ``model.quantize``: the float net as the ``CVSNET01`` bytes ``rust/src/net.rs`` reads.
4. ``store``: the acceptance duel against the built-in objective at equal nodes, and the
   store ``nets/<ruleset-hash>-<procedure>.{net,json}``, named by the procedure and
   carrying the weights' SHA-256 and the build fingerprint.

**Outside every fingerprint set, on purpose.** ``measurement.AGENTS_SOURCES``, the
fallback for a roster whose players cannot be read, is ``agents/*.py``, not this
directory: what a net player plays is its weights, and the basis stamps their digest
(``Basis.net``, and the bytes inside ``Basis.players``). Editing the trainer makes a new net, which is
a new digest; it cannot change a game played by a player without ``net=``.

numpy is the ``train`` extra (``pyproject.toml``). Nothing that plays a game imports
this package.
"""
