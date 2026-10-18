"""What every measurement means, in the language of games rather than of code.

`components.py` computes the numbers and `scoring.py` maps them onto a 0-1 scale. This
says what they are *for*. Everything here is a sentence somebody could act on -- what is
being measured, how it was measured, and what it tells you when it is high or low --
because an index whose components are called `gradient_persistence` and `draw_slope`
is, to anybody who has not read this repository, a column of unlabelled numbers with
one more on top.

Three rules keep this honest.

**Nothing here computes anything.** A glossary that recomputed a value could disagree
with the value being explained, which is a worse failure than not explaining it. Every
number the page shows comes from the index file on disk; this supplies only prose.

**It describes a measurement, not a target.** "Higher is better" is a property of the
curve in a versioned weights file and is read from there, never asserted here. That
distinction is load-bearing: `mean_branching` and `progress_irreversibility` are bands,
and a glossary that called either "more is better" would be telling a reader to
optimise something no player wants -- which is the thing the weights files exist to
prevent.

**It is never shown to the proposing model.** The rule that the proposer sees prose and
never the index is what stops it optimising the proxy, and this module is the index
stated in prose -- which would be the leak in its purest form. `tests/test_loop.py`
holds the brief and the feedback channel to that; nothing in `chess_variant_sandbox/llm/` imports
this, and nothing should.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.metrics import scoring
from chess_variant_sandbox.metrics.scoring import DEFAULT_WEIGHTS, Weights, load_weights
from chess_variant_sandbox.metrics.seat import SEAT_GATE, TOP_THREE_SEAT_GATE


@dataclass(frozen=True)
class Measurement:
    """One raw number, said three ways."""

    # A name a person would use. The component name is the key.
    title: str
    # What it is, in one sentence.
    what: str
    # How it was arrived at, specifically enough to argue with.
    how: str
    # What the number is telling you. Deliberately two-sided for a band, because the
    # direction is the curve's business and not the measurement's.
    reading: str
    # "0-1", "plies", "Elo per doubling", ... Empty where the number is a bare ratio
    # whose scale the prose already carries.
    units: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "what": self.what,
            "how": self.how,
            "reading": self.reading,
            "units": self.units,
        }


# What a family is asking. The index is a weighted *geometric* mean across these rather
# than an arithmetic one over components, and the reason is worth stating wherever the
# families are shown: they are not substitutable. Elegance cannot make up for having no
# strategy, and arithmetic lets it -- the deliberately dull control scored 0.64 that way.
FAMILIES: dict[str, str] = {
    "elegance": "How much game these rules buy. Measured as yield rather than as "
                "thinness: scoring a short rulebook directly rewards having less game "
                "in the box.",
    "balance": "Whether the two sides are playing the same game, and whether the game "
               "reaches an ending at all.",
    "pacing": "How long a game runs, how wide the choices are, and how much of what "
              "happens cannot be undone.",
    "variety": "Whether two strong players can play this twice without playing the "
               "same game twice.",
    "drama": "Whether a lead means anything, whether one mistake ends it, and whether "
             "the game stays decisive as the players get better.",
    "decisions": "Whether there is anything to think about: how much of the move list "
                 "matters, and how much of the army is doing something.",
    "depth": "Whether thinking pays, and whether it keeps paying. The question "
             "everything here exists to answer.",
}


# Keyed by component name. Covers every component any weights version scores, every
# component a gate reads, and the raw measurements that are only ever diagnostics --
# because a page showing the raw block shows all three and a reader cannot tell which
# is which without being told.
MEASUREMENTS: dict[str, Measurement] = {
    # ------------------------------------------------------------------ elegance
    "rule_mass": Measurement(
        title="rule mass",
        what="How much a new player has to be told before they can play.",
        how="A weighted count over the spec: piece types, move descriptors, "
            "conditional moves, and each special rule at a cost reflecting how long "
            "it takes to state. Chess measures about 52.",
        reading="A bigger number is a longer explanation, and it is a fact about the "
                "rules rather than a verdict on them. Only the two oldest versions "
                "scored it on its own, and dropped it because low mass by itself only "
                "means there is less game here. The versions that score rule yield "
                "divide by it.",
        units="weighted count",
    ),
    "piece_types": Measurement(
        title="piece types",
        what="How many kinds of piece the army has.",
        how="Counted off the spec.",
        reading="Context for everything else; never scored.",
        units="count",
    ),
    "special_rules": Measurement(
        title="special rules",
        what="How many exceptions the game carries beyond how its pieces move.",
        how="Castling, en passant, promotion, drops, explosions, regions, a facing "
            "rule -- each weighted by how long it takes to state. It is also the "
            "sixth axis of the behaviour grid.",
        reading="Context for everything else; never scored on its own.",
        units="weighted count",
    ),
    "rule_yield": Measurement(
        title="rule yield",
        what="What the rules bought: how much separation between players each unit of "
             "rulebook produced.",
        how="Rating spread divided by rule mass.",
        reading="High means a lot of game from a little explanation. This replaced "
                "scoring low rule mass directly, which rewarded thinness -- mass "
                "correlates +0.88 with branching and +0.82 with rating spread, so a "
                "lower-is-better curve on it subtracted from the depth signal and put "
                "both deliberately broken controls at the top of the family.",
        units="Elo per unit of rule mass",
    ),
    # ------------------------------------------------------------------- balance
    "first_player_advantage": Measurement(
        title="first-player advantage",
        what="How much of an edge moving first is worth.",
        how="The first seat's mean score in self-play only -- the one matchup where "
            "the two players are identical, so any difference is the seat -- doubled "
            "and taken as a distance from even.",
        reading="0 is a dead-even seat, 1 is a game the first player always wins. The "
                "weakest-measured thing in the index: its spread across real games is "
                "only 1.9 times its own sampling noise, which is why it carries little "
                "weight.",
    ),
    "first_player_score": Measurement(
        title="first-player score",
        what="What the first seat actually scores, before it is folded into a distance.",
        how="Mean score for seat 0 across self-play games; 0.5 is even.",
        reading="Shows which way the advantage runs, which the folded number cannot.",
    ),
    "seat_decided_chance": Measurement(
        title="seat-decided chance",
        what="How likely a fair seat is to produce the split the two strongest players' "
             "games showed.",
        how="Over every game whose two seats are both in the top two rungs of the ladder "
            "-- both self-play pairings and both orders of the cross pairing, 16 games on "
            "the canonical roster -- the first seat's wins and losses go into a two-sided "
            "exact sign test, each different decided game counted once however often its "
            "players replayed it, since a line played sixteen times is one observation of "
            "what its pairing does (Q953a). Draws are dropped.",
        reading="A gate rather than a component: below 0.001 the ruleset is disqualified, "
                "because a game the strongest players win from one seat nearly every "
                "time, whoever sits in it, is decided by the seat and not by play. On 16 "
                "games only a clean or near-clean sweep reaches that; a lean is read by "
                "'seat-decided chance, top three' beside it. Shown rounded; the exact "
                "figure and the split are in the diagnostics. This figure is one run's. "
                "A library game is measured four times and its seat is decided once over "
                "the four runs' games together, against a tolerated share of two in "
                "three rather than against an even seat, so that what the check means "
                "stops moving as more runs are added.",
        units="probability",
    ),
    "top_rung_first_seat_score": Measurement(
        title="first-seat score, top rungs",
        what="What the first seat scores in the games between the two strongest players.",
        how="Mean score for seat 0 over the same games the seat-decided chance reads; "
            "0.5 is even.",
        reading="Shows which way a refused split runs. Never scored, and on 16 games too "
                "wide to grade anything narrower than a sweep.",
    ),
    "top_three_seat_decided_chance": Measurement(
        title="seat-decided chance, top three",
        what="The same question as the seat-decided chance, over the games among the "
             "three strongest players instead of the two.",
        how="Over every game whose two seats are both in the top three rungs of the "
            "ladder -- 36 games on the canonical roster -- the first seat's wins and "
            "losses go into the same two-sided exact sign test, each different decided "
            "game counted once. Draws are dropped. A "
            "roster whose third rung searches nothing, as the design loop's does, emits "
            "no reading here and is judged on the pair alone.",
        reading="A gate rather than a component: below 0.0005 the ruleset is "
                "disqualified. Three times the games lets this one see a strong lean "
                "and not only a sweep -- 29 of 36, or 25 of 29 decided -- which is what "
                "a game whose first mover wins nine times in ten looks like. A library "
                "game reads this band over its four runs pooled, against the same "
                "tolerated share the pair is read against.",
        units="probability",
    ),
    "top_three_first_seat_score": Measurement(
        title="first-seat score, top three",
        what="What the first seat scores in the games among the three strongest players.",
        how="Mean score for seat 0 over the same games the top-three seat-decided chance "
            "reads; 0.5 is even.",
        reading="Shows which way a refused split runs. Never scored.",
    ),
    "stall_rate": Measurement(
        title="stall rate",
        what="How often a game fails to reach a real ending and is stopped by a clock.",
        how="The share of games between competent players ending on the ply limit or "
            "the no-progress clock. Not repetition: a game that repeats has reached an "
            "ending its own rules name, and counting it here was a mis-statement this "
            "entry carried until 2026-09-07.",
        reading="Low is better, and this is the number the score reads. The gate reads "
                "the worst half of the budget ladder instead -- see 'stall rate, worst "
                "half'.",
    ),
    "stall_rate_worst_half": Measurement(
        title="stall rate, worst half",
        what="The stall rate on whichever half of the budget ladder stalls more.",
        how="Games between competent players are split by their *weaker* seat's node "
            "budget -- can both players see it? -- at the median rung of the graded "
            "ladder, and the higher of the two halves' stall rates is reported.",
        reading="A gate rather than a component: above 0.7 the ruleset is disqualified. "
                "Half a ladder that mostly fails to finish is not rescued by the other "
                "half finishing, which is what a pooled rate lets it be.",
    ),
    "decisive_rate": Measurement(
        title="decisive rate",
        what="How often somebody actually wins.",
        how="The share of games between competent players with a winner.",
        reading="Read as a diagnostic here; the gate reads the worst half of the ladder "
                "-- see 'decisive rate, worst half'.",
    ),
    "decisive_rate_worst_half": Measurement(
        title="decisive rate, worst half",
        what="The decisive rate on whichever half of the budget ladder wins less often.",
        how="Games between competent players are split by their weaker seat's node "
            "budget at the median rung of the graded ladder, and the lower of the two "
            "halves' decisive rates is reported.",
        reading="A gate rather than a component: below 0.2 the ruleset is disqualified. "
                "A game competent players almost always draw is not a contest, and "
                "averaging in the games their weaker opponents lost outright does not "
                "make it one. The dull control is thrown out here.",
    ),
    # -------------------------------------------------------------------- pacing
    "mean_plies": Measurement(
        title="game length",
        what="How long a game lasts.",
        how="Mean plies -- single moves, so a 40-move chess game is 80 plies -- across "
            "games between competent players.",
        reading="Read as a diagnostic here; the gate reads the worst half of the ladder "
                "-- see 'game length, worst half'.",
        units="plies",
    ),
    "mean_plies_worst_half": Measurement(
        title="game length, worst half",
        what="Mean game length on whichever half of the budget ladder finishes fastest.",
        how="Games between competent players are split by their weaker seat's node "
            "budget at the median rung of the graded ladder, and the shorter of the two "
            "halves' mean lengths is reported.",
        reading="A gate rather than a component: under 8 plies the ruleset is "
                "disqualified, because a game decided before either player has "
                "developed anything is not a game. Pooled with a shuffle at the other "
                "end of the ladder, a mean length describes no game that was played -- "
                "the broken control averages 118 plies out of a five-ply trap and a "
                "213-ply grind.",
        units="plies",
    ),
    "ply_spread": Measurement(
        title="length variety",
        what="Whether games vary in length or all run about the same.",
        how="Coefficient of variation of game length: the standard deviation divided "
            "by the mean, so it is comparable between a long game and a short one.",
        reading="Higher means games differ from each other. A game that always ends on "
                "the same ply is running a script.",
    ),
    "mean_branching": Measurement(
        title="choices per turn",
        what="How many legal moves a player typically has.",
        how="Mean legal move count over every ply of every graded game.",
        reading="A band, not a direction: too few and there is nothing to decide, too "
                "many and the position is soup. Also a gate at the bottom -- under 1.5 "
                "there is no play at all.",
        units="moves",
    ),
    "quick_kill_rate": Measurement(
        title="quick kills",
        what="How often the game is over before it has started.",
        how="The share of graded games finishing inside 12 plies with a winner.",
        reading="Read as a diagnostic here; the gate reads the worst half of the ladder "
                "-- see 'quick kills, worst half'.",
    ),
    "quick_kill_rate_worst_half": Measurement(
        title="quick kills, worst half",
        what="The quick-kill rate on whichever half of the budget ladder falls into the "
             "trap.",
        how="Games between competent players are split by their weaker seat's node "
            "budget at the median rung of the graded ladder, and the higher of the two "
            "halves' quick-kill rates is reported.",
        reading="A gate rather than a component: above 0.5 the ruleset is disqualified. "
                "A game where half of all contests end in a handful of moves is decided "
                "by an opening trap rather than by play. This is what catches bastion's "
                "three-ply forced mate: every quick kill in that run is a game one seat "
                "was too weak to see it coming in, 80 of 132, and pooling them with the "
                "64 games the stronger seats ground out is what let the same 80 kills "
                "read as 0.408 and pass.",
    ),
    "progress_irreversibility": Measurement(
        title="irreversible progress",
        what="How much of what happens can never be undone by playing on.",
        how="The share of plies that reset the no-progress clock -- a capture, a move "
            "by a one-way piece, a promotion, or a permanent claim on a square.",
        reading="A band. Too low and the game can shuffle forever with nothing but its "
                "own clock to stop it, which is what a designer watching his own game "
                "diagnosed as 'the gameplay gets very repetitive'. Too high and it is "
                "over in a handful of moves: the two worst games in the corpus measure "
                "1.000.",
    ),
    # ------------------------------------------------------------------- variety
    "opening_diversity": Measurement(
        title="opening variety",
        what="Whether strong players open differently from game to game.",
        how="Normalised entropy over the first few moves of games between competent "
            "players. Random movers are excluded: a random mover produces endless "
            "variety in any ruleset whatsoever, so including them would measure the "
            "dice rather than the game.",
        reading="1.0 is every opening equally likely; low means one line is forced.",
    ),
    "distinct_game_rate": Measurement(
        title="distinct games",
        what="How often two games between the same players are actually different games.",
        how="The share of graded games whose move sequence is unique in the corpus. "
            "Each game counts once, so a bootstrap resample that draws one game twice "
            "does not read the copy as a repeat.",
        reading="1.0 is no repeats. Low means the same game keeps being replayed, which "
                "is the failure a player notices first.",
    ),
    "position_divergence": Measurement(
        title="games that go different places",
        what="How far each game strong players produced is from the most similar other "
             "one, judged by the positions the two passed through.",
        how="From the analysis pass: each graded game's set of positions, and one minus "
            "the share it has in common with its nearest other game (shared positions "
            "over positions either saw), averaged over the games. Each game counts once "
            "as a neighbour, so a bootstrap resample that draws one game twice does not "
            "find a copy at distance zero.",
        reading="Near 1 the games part early and never meet again; near 0 the same game "
                "keeps being replayed. Two games that reach the same positions in a "
                "different order read as the same game, which a move list cannot see. "
                "Long games read higher, because two long games rarely share a position.",
    ),
    "event_divergence": Measurement(
        title="games that do different things",
        what="As games that go different places, but reading only the positions reached "
             "by a move that cannot be undone.",
        how="From the analysis pass: the positions after each capture, one-way move, "
            "promotion or permanent claim, compared between each graded game and its "
            "nearest other game. A game with no such move scores zero and is nobody's "
            "neighbour.",
        reading="Near 1 the games did different irreversible things; near 0 they did the "
                "same ones. Pieces wandering between the same events add nothing here. "
                "A game whose moves are all reversible reads zero by definition, which "
                "is a fact about the measurement and not a verdict on the design.",
    ),
    "prefix_divergence": Measurement(
        title="games that part early",
        what="How much of each game its most similar other game plays move for move.",
        how="One minus the share of the shorter game that two games play identically "
            "from the first move, to the nearest other graded game, averaged over the "
            "games. Read off the move lists alone.",
        reading="Measured but not scored: a comparison for the position form. It reads "
                "two games that transpose into the same positions as different games, "
                "and a tie broken differently on the first move as a whole different "
                "game.",
    ),
    # --------------------------------------------------------------------- drama
    "lead_changes": Measurement(
        title="lead changes",
        what="How often the material advantage crosses from one player to the other.",
        how="Counted per hundred plies from the material series every record carries.",
        reading="Higher is swingier. Not scored in the current weights -- it opened its "
                "'good' band at 4.0 where chess measures 1.96, which was one of the "
                "defects the v3 re-anchoring was built to fix.",
        units="per 100 plies",
    ),
    "material_volatility": Measurement(
        title="material volatility",
        what="How violently the material balance moves.",
        how="Mean absolute change in material per ply.",
        reading="Context rather than a component: it separates a game of constant "
                "trading from one of long manoeuvring, and neither is better.",
        units="material per ply",
    ),
    "draw_slope": Measurement(
        title="decisiveness under skill",
        what="Whether the game stays decisive as the players get better.",
        how="The decisive rate among the strongest searchers divided by the same among "
            "the weakest.",
        reading="A band around 1.0. Below 1 the game drifts towards drawn theory as "
                "skill rises, which is Capablanca's complaint about chess; far above 1 "
                "means weak players cannot finish it.",
    ),
    "blunder_recoverability": Measurement(
        title="recovery from a mistake",
        what="Whether one bad move ends the game.",
        how="From the analysis pass: a forced blunder is played into a real position "
            "and the game is finished from there. This is the share of those trials the "
            "blundering side still went on to save.",
        reading="A band. At 0 a single error is fatal and the game is a memory test; at "
                "1 nothing you do matters. Noisy -- about 2.5 times its own sampling sd "
                "-- but it separates broken games from real ones harder than almost "
                "anything else here.",
    ),
    "lead_conversion": Measurement(
        title="converting a lead",
        what="Whether being ahead is worth anything.",
        how="Among decisive games where one side held a real material edge partway "
            "through, how often that side went on to win.",
        reading="0.5 says a lead is worth nothing. Optional: a short run may simply not "
                "contain enough decisive games carrying a real edge, and that absence "
                "is recorded rather than scored as zero. Withheld entirely where the "
                "ladder says the roster cannot read the ruleset, because material is "
                "the wrong objective in antichess and a number describing the evaluator "
                "must not be reported as a property of the design.",
    ),
    # ----------------------------------------------------------------- decisions
    "decision_criticality": Measurement(
        title="decisions that matter",
        what="How much of the move list would actually cost you something.",
        how="From the analysis pass: every legal move in a sampled position is scored "
            "at a full window, and this is one minus the share that come out near the "
            "best. The ruler is the same on every game: the built-in evaluator on the "
            "piece values the rules imply, and near means within the cheapest piece. "
            "Where the players use a fitted objective, this still does not, so the "
            "number compares across games but can be blind where the built-in "
            "evaluator is.",
        reading="A band. Near 0 every move is as good as every other and there is "
                "nothing to think about; near 1 there is exactly one move and the game "
                "is a puzzle. The best discriminator in the set against known-bad "
                "games, and the thing choices-per-turn was standing in for -- a move "
                "count cannot tell a wide game from a wide move list with one sane "
                "entry in it.",
    ),
    "piece_utilisation": Measurement(
        title="use of the army",
        what="Whether all the kinds of piece are doing something.",
        how="Normalised entropy over how many moves each piece *type* made, so a game "
            "with three types and a game with nine are comparable. It counts the types "
            "a player can hold (placed, stocked, promoted to, or taken into a hand), so "
            "a wild piece nobody owns is not charged, nor is a portal, and a held type "
            "that never moves still is.",
        reading="1.0 is every type pulling its weight; low means some pieces are "
                "scenery. Its known blind spot is a game that collapses a piece's value "
                "without leaving it idle -- an average entropy barely registers that, "
                "and one of the deliberately broken test games exploits it.",
    ),
    "instance_participation": Measurement(
        title="pieces that move",
        what="Whether the individual pieces are doing something, as opposed to their "
             "kinds.",
        how="From the analysis pass: the share of pieces on the board at the start that "
            "ever moved. A portal is part of the board, not a piece, and is not counted.",
        reading="A band. Use of the army is an entropy over types and cannot see five "
                "rooks sitting in the back while a sixth does all the work; this counts "
                "instances. It is here because a designer playtesting a game the "
                "library had never seen named the failure, and named it per piece.",
    ),
    "capture_availability": Measurement(
        title="captures on offer",
        what="Whether the two armies ever meet.",
        how="From the analysis pass: the mean share of legal moves that would take "
            "something.",
        reading="Measured but deliberately not scored: it is the best-measured "
                "candidate of its cohort and correlates -0.84 with use of the army, so "
                "paying for both would be counting one measurement twice. A near-zero "
                "here is two solitaires sharing a table.",
    ),
    "position_revisit_rate": Measurement(
        title="repeated positions",
        what="How much of a game is spent in positions it has already been in.",
        how="From the analysis pass: the share of plies whose position key had been "
            "seen before in that game.",
        reading="Measured but not scored: its spread across real games is narrower than "
                "the minimum curve width the anchoring rules allow, so any compliant "
                "curve would be flat across the whole corpus. A repetition *rule* only "
                "catches the third occurrence; this catches a game that spends its life "
                "in a dozen positions without ever tripping it.",
    ),
    # --------------------------------------------------------------------- depth
    "skill_gradient": Measurement(
        title="skill gradient",
        what="Whether thinking pays: what doubling a player's thinking budget is worth.",
        how="A least-squares fit of Elo against log2 of the node budget, across the "
            "searching agents on the ladder.",
        reading="Higher means skill exists here. Also a gate: under 8 Elo per doubling "
                "the ruleset is disqualified, because if more thought buys nothing "
                "there is nothing to be good at. Chess measures about 120.",
        units="Elo per doubling",
    ),
    "gradient_persistence": Measurement(
        title="does thinking keep paying",
        what="Whether more thought still helps once the players are already strong.",
        how="Elo per doubling across the top half of the budget ladder, over the same "
            "across the bottom half.",
        reading="A band around 1. The distinction a single gradient cannot draw: both a "
                "deep game and a nearly solved one reward search at modest budgets, and "
                "only the deep one still rewards it at the top. Needs six budgets to "
                "mean anything -- with four the two halves shared a point.",
    ),
    "rating_spread": Measurement(
        title="rating spread",
        what="How far apart the best and worst players on the ladder end up.",
        how="The Elo range of the fitted Bradley-Terry ratings.",
        reading="Not scored directly -- it correlates +0.93 with the skill gradient, so "
                "weighting both would have expressed one measurement twice, which is "
                "what v2 did with 0.22 of its weight. It survives as the numerator of "
                "rule yield.",
        units="Elo",
    ),
    "search_advantage": Measurement(
        title="search beats chance",
        what="Whether a player that thinks beats one that does not.",
        how="The score the strongest agent takes off a random mover, on a 0-1 scale.",
        reading="A gate rather than a component: under 0.55 the ruleset is "
                "disqualified. It was scored in v2 at 0.06 of the weight and has zero "
                "variance across every real game, which is the definition of a "
                "component paying for nothing.",
    ),
    "ladder_monotonicity": Measurement(
        title="is the ladder a ladder",
        what="Whether a bigger thinking budget actually produced a better player.",
        how="The share of adjacent budget steps on which the larger search rated higher.",
        reading="A check on the instrument, never a component. Antichess measures 0.33 "
                "-- its biggest search rates level with a random mover, because the "
                "evaluator maximises material in a game won by shedding it. Every skill "
                "number for such a ruleset describes the agents rather than the game, "
                "and scoring this would be marking a ruleset down for the roster's "
                "inability to read it.",
    ),
    # ------------------------------------------------- read off the spec, not played
    "threat_asymmetry": Measurement(
        title="threat asymmetry",
        what="Whether pieces can strike from where they cannot be struck back.",
        how="Read off the spec before anything is played: the share of ordered pairs of "
            "piece types where one can attack the other from out of its reach.",
        reading="0.000 on a game whose pieces all move alike. It is the only thing here "
                "that could tell a non-transitive capture game from a uniform one, "
                "which is why the obvious 'the pieces all move the same, so it is dull' "
                "component was rejected: rock, paper and scissors all move like kings, "
                "and the asymmetry is carried by what they may take.",
    ),
    "traversal_time": Measurement(
        title="crossing time",
        what="How long it takes a piece to cross the board.",
        how="Board diameter divided by how far a typical piece travels in one move.",
        reading="A diagnostic and a candidate grid axis, never scored: asserting that a "
                "slower game is a better one is a taste claim, not a measurement.",
        units="moves",
    ),
    # ----------------------------------------- the heuristic pass, none of them scored
    "heuristic_headroom": Measurement(
        title="heuristic headroom",
        what="What thinking harder buys, when both players want exactly the same thing.",
        how="A one-ply player and a deep search are given the *same* stated plan -- the "
            "same weights over the same measurements -- and play each other. The number "
            "is what the deep side won by, rescaled so that 'depth changed nothing' is "
            "zero. Reported as the median over the set of plans; playing the deep side "
            "against a plan it cannot beat is a fact about the search, so a loss reads "
            "as zero rather than going negative.",
        reading="Low says a rule of thumb plays this as well as a search of the same "
                "objective. High says there is something to work out. It is not a "
                "verdict on the design in either direction: the deliberately broken "
                "control hides a three-ply forced win and therefore reads high, and a "
                "game whose deep arm never got deep is withheld rather than called "
                "shallow. Never scored.",
    ),
    "heuristic_agreement": Measurement(
        title="objective disagreement",
        what="Whether some stated plan out-plays the evaluator every rating is built on.",
        how="The best plan against the built-in evaluator, at one ply and at the deep "
            "budget, taking the worse of the two. It has to win at both depths before "
            "this claims anything.",
        reading="Above a half means the built-in evaluator is pointed at the wrong thing "
                "on this ruleset, and every strength number for it carries an asterisk. "
                "This is a statement about the instrument rather than about the game -- "
                "the same category as the ladder check -- except that it also names what "
                "the evaluator should have been maximising instead. Never scored.",
    ),
    "heuristic_depth_reached": Measurement(
        title="depth reached",
        what="How many plies the deep side of the headroom comparison actually saw.",
        how="The deepest completed iteration of deepening, averaged over the opening "
            "twelve plies, at the budget chosen for this ruleset.",
        reading="A control rather than a measurement of the game. On a wide board a "
                "fixed budget buys two plies, and a two-ply 'deep' arm would make every "
                "game look shallow for a reason that has nothing to do with the rules. "
                "Where this stays too low after the budget has been raised, headroom is "
                "withheld. Never scored.",
        units="plies",
    ),
}


# What a visitor to the site reads for a measurement, where it is not the maintainer's
# sentence above (`docs/archive/plan-understanding.md` §3.3, rows A7, A19 and A20). Keyed like
# `MEASUREMENTS`; each entry replaces `title`, `what`, `reading` or `units` and nothing
# else. The `how` line is the maintainer's alone, so the site draws none. `explain` sends
# these beside the maintainer's words as a component's `site`, and the page draws them
# where it is the site (`ACCOUNTS`), as `candidacy.SITE_WORDS` does for the designer's
# steps. Words only: no number here is a measurement's, and none is computed.
SITE_WORDS: dict[str, dict[str, str]] = {
    "rule_mass": {"title": "complexity"},
    "rule_yield": {"units": "rating points per unit of complexity"},
    "first_player_advantage": {
        "reading": "0 is no edge at all, 1 is a game the first player always wins. It is the "
                   "least certain thing measured: games differ on it by not much more than "
                   "chance alone moves it, which is why it counts for little.",
    },
    "stall_rate": {
        "title": "games that never finish",
        "reading": "Low is better. The test that rules a game out reads it on whichever half "
                   "of the computer players, weaker or stronger, does worse.",
    },
    "mean_plies": {"units": "turns"},
    "mean_plies_worst_half": {"units": "turns"},
    "ply_spread": {
        "reading": "Higher means games differ from each other. A game that always ends on "
                   "the same turn is running a script.",
    },
    "mean_branching": {
        "reading": "Best in the middle: too few and there is nothing to decide, too many and "
                   "the position is soup. Under 1.5 there is no play at all, and the game is "
                   "ruled out.",
    },
    "lead_changes": {"units": "per 100 turns"},
    "material_volatility": {"units": "material per turn"},
    "blunder_recoverability": {
        "reading": "Best in the middle. At 0 a single error is fatal and the game is a memory "
                   "test; at 1 nothing you do matters. Chance moves it a lot, but it tells "
                   "broken games from real ones better than almost anything else here.",
    },
    "lead_conversion": {
        "reading": "0.5 says a lead is worth nothing, and too high says the first real lead "
                   "decides the game. A run may not have enough won games with a real lead "
                   "in them, and then this is left out rather than scored as zero. It is "
                   "also left out where the computer players can't judge who is ahead here, "
                   "because a number about them must not be read as a fact about the game.",
    },
    "decision_criticality": {
        "reading": "Best in the middle. Near 0 every move is as good as every other and "
                   "there is nothing to think about; near 1 there is exactly one move and the "
                   "game is a puzzle. It tells broken games from real ones better than "
                   "anything else measured: a count of moves cannot tell many good choices "
                   "from many moves with one sensible one among them.",
    },
    "piece_utilisation": {
        "reading": "1 is every kind of piece pulling its weight; low means some pieces are "
                   "scenery. It can miss a game that makes a piece worthless without leaving "
                   "it idle, and one deliberately broken test game does exactly that.",
    },
    "instance_participation": {
        "reading": "Best in the middle. Use of the army looks at kinds of piece, and cannot "
                   "see five pieces of one kind sitting at the back while a sixth does all "
                   "the work; this counts each piece. It is here because a designer testing "
                   "a new game named that failure.",
    },
    "skill_gradient": {
        "title": "does thinking longer win",
        "what": "Whether thinking pays: what a computer player gains each time it thinks "
                "twice as long.",
        "reading": "Higher means skill exists here. Under 8 rating points each time thinking "
                   "time doubles, the game is ruled out, because if more thought buys nothing "
                   "there is nothing to be good at. Chess measures about 120.",
        "units": "rating points each time thinking time doubles",
    },
    "rating_spread": {"units": "rating points"},
    "heuristic_depth_reached": {"units": "turns"},
}


def explain(version: str = DEFAULT_WEIGHTS) -> dict[str, Any]:
    """The glossary joined to the weights actually in force.

    Two halves that have to be shown together and are stored apart on purpose. What a
    measurement *is* does not change between versions; what it is worth, and which way
    the curve runs, is exactly what a version is. Reading the second from the weights
    file rather than restating it here is what stops a glossary from quietly describing
    a scoring rule that no longer exists.
    """
    weights = load_weights(version)
    roster_rho = _roster_rho(weights.version)
    return {
        "version": weights.version,
        "notes": weights.notes,
        "families": {
            name: {"means": text, "weight": round(weight, 4)}
            for name, weight in sorted(weights.family_weights().items())
            for text in [FAMILIES.get(name, "")]
        },
        "components": {
            name: _component(name, weights, roster_rho) for name in sorted(weights.weights)
        },
        "gates": [
            {
                "component": gate.component,
                "title": title_of(gate.component),
                "minimum": gate.minimum,
                "maximum": gate.maximum,
                "reason": gate.reason,
            }
            for gate in (*weights.gates, SEAT_GATE, TOP_THREE_SEAT_GATE)
        ],
        "measurements": {
            name: measurement.as_dict() for name, measurement in MEASUREMENTS.items()
        },
        "howItWorks": HOW_IT_WORKS,
        "howItWorksSite": SITE_HOW_IT_WORKS,
    }


def _component(name: str, weights: Weights, roster_rho: dict[str, float]) -> dict[str, Any]:
    curve = weights.curves[name]
    return {
        "family": weights.families.get(name, "other"),
        "weight": round(weights.weights[name], 4),
        "optional": name in weights.optional,
        "curve": {"kind": curve.kind, "points": dict(curve.points)},
        **(MEASUREMENTS[name].as_dict() if name in MEASUREMENTS else {}),
        "site": SITE_WORDS.get(name, {}),
        **({"roster_rho": roster_rho[name]} if name in roster_rho else {}),
    }


def _roster_rho(version: str) -> dict[str, float]:
    """A version's own ``roster_rho`` per component, read straight off its JSON file.

    Optional (Q52, `docs/archive/plan-variety.md` A3): the rank correlation between two rosters'
    readings of a component's raw measurement, against `docs/reference/roster-confound.md` section
    8's threshold of 0.50. `v10` and every version before it carry none, because the
    field did not exist when they were written; a version from `v11` on may carry it for
    every component it scores.

    **Read off the raw file, not off `Weights`.** `scoring.parse_weights` does not carry
    the field, and it must not gain it: `Weights` lives in `metrics/scoring.py`, which is
    inside the code reading `measurement.METRICS_SOURCES` fingerprints, so adding a field
    there -- even one nothing in that module's own logic uses -- would move the `metrics`
    fingerprint and stale every stored score by re-score. This module is `DISPLAY_ONLY`
    (`measurement.DISPLAY_ONLY`): reading the file a second time, here, to pull out one
    more key is what keeps the field's arrival from moving anything.

    Uses `scoring.WEIGHTS_DIR` rather than a bound import of it, so a test that
    monkeypatches the module attribute to point at a made-up version reaches this too --
    the same file `load_weights` just parsed for the rest of `explain`.

    Malformed or missing is empty rather than raised: `load_weights` above has already
    validated the file enough to build a `Weights` from it, and a display extra is not
    where that validation belongs a second time.
    """
    try:
        data = json.loads((scoring.WEIGHTS_DIR / f"{version}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    components = data.get("components") if isinstance(data, dict) else None
    if not isinstance(components, dict):
        return {}
    found: dict[str, float] = {}
    for name, config in components.items():
        if not isinstance(config, dict):
            continue
        value = config.get("roster_rho")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            found[name] = float(value)
    return found


def title_of(name: str) -> str:
    """A component's readable name, falling back to the identifier with its underscores
    taken out -- which is what a component nobody has written a glossary entry for
    should look like, rather than an empty string."""
    entry = MEASUREMENTS.get(name)
    return entry.title if entry else name.replace("_", " ")


# The three things a reader has to know before the headline number means anything. Kept
# here rather than in the page because they are claims about the method, and the method
# lives in this package.
HOW_IT_WORKS: list[dict[str, str]] = [
    {
        "heading": "It is a gate and then a score",
        "body": "Some conditions disqualify rather than deduct. A game that never "
                "finishes, or that a thinking player cannot beat a random one at, is "
                "not a mediocre game -- letting a strong showing elsewhere average that "
                "away is how a broken ruleset reaches the top of a leaderboard. A gated "
                "game still carries its number, and the number does not mean anything.",
    },
    {
        "heading": "The outcome gates read the worse half of the ladder",
        "body": "Games here are played by a ladder of players at different thinking "
                "budgets, and a ruleset can be two different games on it: an opening "
                "trap against the players too weak to see it, a dead shuffle against "
                "the ones who can. Averaged over the whole ladder each failure is the "
                "other one's alibi -- the trap games look decisive, the shuffle games "
                "look long -- and both gates pass on a mean length no game ever had. So "
                "the four outcome gates split the games at the ladder's midpoint and "
                "read whichever half is worse. The scores still read the whole set.",
    },
    {
        "heading": "Families multiply, they do not average",
        "body": "Components are averaged inside a family and the families are combined "
                "by a weighted geometric mean, so a family scoring near zero drags the "
                "whole index down instead of being outvoted. Arithmetic was tried first "
                "and the deliberately dull control scored 0.64 with it: simplicity, "
                "variety and drama all max out trivially on a game with nothing in it.",
    },
    {
        "heading": "The interval is wider than the gaps, not most of them",
        "body": "Every score is resampled from the games actually played, and the "
                "resulting standard deviation is about 0.022 against a band of real "
                "games 0.42 wide. That sounds like room to rank in and it is not: as "
                "of the 2026-09-26 replay, only 4 of the library board's 59 adjacent "
                "pairs have intervals that clear each other. The index separates a "
                "game from a non-game and does not rank games -- read a number as a "
                "band, and read an ordering as noise unless the intervals clear each "
                "other.",
    },
    {
        "heading": "The interval describes the runs it came from",
        "body": "A run is one tournament from one seed. For a score taken over one run "
                "the interval says how much that run's score moves when its own games "
                "are resampled, corrected for reading four games a pairing as though "
                "they were the whole population. It is not a forecast of a replay: "
                "playing the same game again draws a fresh set of games and carries a "
                "second run's error, and four replays of the whole library put a fresh "
                "run within four of these standard deviations about nineteen times in "
                "twenty.",
    },
    {
        "heading": "A library score is the mean of four runs",
        "body": "Every game kept here is played from four seeds and scored four times, "
                "and the score is the mean of the four totals -- never one score over "
                "the four runs' games pooled, which is a different quantity and would "
                "need its own weights version. Its interval combines the four runs' own "
                "half-widths, so it is the sampling error of the mean and about half as "
                "wide as one run's. When the library was played again from four more "
                "seeds, the new mean fell inside it about five times in six, which is "
                "what an honest interval should do, because the new mean carries its "
                "own error too. A game measured once carries "
                "one run and says so, and a mean needs three of the four runs to have "
                "produced a total at all.",
    },
    {
        "heading": "A version is a different quantity",
        "body": "Weights are versioned and never edited; a re-anchoring is a new "
                "version. A score records the version that produced it, and two scores "
                "under different versions must not be compared.",
    },
]


# The same account in a visitor's words (`docs/archive/plan-understanding.md` §3.3, A22), drawn by
# the page where it is the site, as `explain`'s `howItWorksSite`. One entry for each of
# `HOW_IT_WORKS`', in the same order, saying the same thing without the maintainer's terms
# (*gate*, *ladder*, *weighted geometric mean*, *pooled*, *half-widths*, *re-anchoring*)
# and without the dated calibration figure, which the maintainer's version keeps.
SITE_HOW_IT_WORKS: list[dict[str, str]] = [
    {
        "heading": "Some failings rule a game out",
        "body": "A game that never finishes, or where a player who thinks is no better than "
                "one who moves at random, is not a mediocre game, and a strong showing "
                "elsewhere must not make up for it. So it is ruled out rather than marked "
                "down. A game that is ruled out still shows its score, and the score means "
                "nothing.",
    },
    {
        "heading": "Those tests read the half that does worse",
        "body": "The computer players come in several strengths, and one set of rules can "
                "be two different games among them: an opening trap against the players "
                "too weak to see it, a dead shuffle against the ones who can. Averaged "
                "together, each failing hides the other. So the four tests on how games "
                "end split the games between the weaker and the stronger computer players "
                "and read whichever half does worse. The score still reads every game.",
    },
    {
        "heading": "The qualities are multiplied together",
        "body": "The things measured are averaged within each quality, and the qualities "
                "are multiplied together, so one quality near zero pulls the whole score "
                "down instead of being outvoted. Averaging was tried first, and a "
                "deliberately dull test game scored 0.64 with it: it looked simple, varied "
                "and dramatic only because there was nothing in it.",
    },
    {
        "heading": "Most games are too close to rank",
        "body": "Every score has a range from chance alone, and the ranges are wider than "
                "most of the gaps between games. So most games are too close to rank: read "
                "a score as rough, never as a place. The score tells a working game from a "
                "broken one; it does not put working games in order.",
    },
    {
        "heading": "The range describes the games it came from",
        "body": "A run is one tournament of a few hundred games among the computer "
                "players. For a score from one run, the ± says how much the score moves "
                "when that run's own games are drawn again at random. It is not a forecast: "
                "playing the game again gives fresh games, so a new run can land further "
                "away than the ±, though nineteen times in twenty it lands within four "
                "times it.",
    },
    {
        "heading": "A library score is the average of four runs",
        "body": "Every game kept in the library is measured four times, on four different "
                "sets of games, and its score is the average of the four scores, never one "
                "score over all their games at once. Its ± comes from the four runs "
                "together, so it is about half as wide as one run's. When the whole library "
                "was measured again on four more sets of games, the new average landed "
                "inside it about five times in six, which is what an honest ± should "
                "do, because the new average has its own error too. A game measured once "
                "says 1 run, and an average needs three of the four runs to have a score "
                "at all.",
    },
    {
        "heading": "Each version of the scoring rules is its own",
        "body": "The scoring rules have numbered versions, and scores under different "
                "versions are never compared. A change to the rules is a new version, never "
                "an edit, and every score says which version produced it.",
    },
]
