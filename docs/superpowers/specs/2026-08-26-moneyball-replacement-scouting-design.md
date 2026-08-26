# Moneyball replacement scouting — design

Date: 2026-08-26. Status: draft for review.

## 1. Goal

Help a smaller club in Europe's Big 5 leagues replace a player who is being bought out: find
someone who delivers the same or more contribution *in that club's system*, for far less than the
market charges for the obvious name. The scout's question is **"We're losing X. Who replaces him?"**

The answer is a ranked shortlist. Each candidate (1) fills X's role in the club's shape, (2)
matches how the club plays, (3) has opponent- and league-adjusted contribution at or above X's,
(4) is priced well below what X's replacement "should" cost, and (5) has trajectory on his side.
Every number carries an interval and a reason. The replacement is fitted to the club, not cloned
from X; the model may conclude X was a poor fit who happened to be good.

### Origin

This project grows out of a first-course clustering exercise (per-category k-means on 460 Big-5
attackers, 2024-25, co-occurrence similarity). That work is kept under `legacy/` as the documented
origin and baseline. Its similarity idea survives as one component (§4.H), not as the engine.

### Success criteria

- In the departure backtest (§5), the model's shortlist beats the club's actual signing, a naive
  rule, and the market's own ranking on model-independent outcomes per euro (minutes share, raw
  goals+assists per 90, realised value change) in most backtest seasons, with per-season
  intervals shown.
- Every component passes, or is reported as failing, its kill check (§5.3).
- Every number in the writeup traces to a committed artifact; `make train` reproduces them from a
  clean clone and in Docker.
- The writeup quotes ranges, not point estimates; reports negative and withdrawn findings with
  the checks that killed them; and states plainly the share of outcome variance the models do
  *not* explain.

## 2. Scope

### Tier 1 — Big 5 (evaluation)

England, Spain, Italy, Germany, France; seasons 2014-15 onward — the full range of the only
verified multi-season xG-grade source (Understat). Full model and full backtest live here. League
conversion factors are estimated from players who moved into the Big 5.

Why all seasons: the backtest needs each season as an independent test (≈10 windows); trajectories
need the same player across years; the transfer panel (~150-250 usable Big-5 moves per season)
needs a decade to support per-role estimates. Market inflation, the 2020-21 valuation dip and xG
model drift are handled by season effects, not by trimming.

### Tier 2 — feeder leagues (candidate pool)

Leagues chosen from the Transfermarkt transfer table: the top source leagues by count of signings
made by smaller Big-5 clubs since 2014. Players there are scored with the same components,
converted to the buyer's league, and shown with wider intervals. A feeder league enters only if
the spike (§7, Phase 0) finds usable player stats for it; leagues with 5+ seasons also contribute
to conversion-factor estimation, leagues with fewer are candidates only.

### Positions

All outfield roles, defined from per-match position codes (Understat: e.g. `AML`, `ML`, `FWL`,
`DMC`), not from season labels. Minutes by role per season. Goalkeepers are conditional: a
goals-prevented proxy (goals conceded vs xG of on-target shots faced, from shot-level data) is
tested for season-to-season stability in Phase 0/2; if unstable, keepers are cut with that evidence.

### Out of scope, with reasons

- Formation strings as features: listed formations are unreliable; the information that matters
  (role) is in per-match position codes.
- Event-level data (StatsBomb open data): no continuous Big-5 seasons.
- Own xG model: Understat's is adequate and not the question.
- Deep learning: thousands of rows, not millions; hierarchical models are stronger and explainable.
- Binary "big game vs easy game" splits: replaced by a continuous opponent-strength adjustment.

### Conditional items (resolved by Phase 0)

| Item | Enters if |
|---|---|
| Goalkeepers | proxy metric is stable year to year |
| Style fit in the ranking | FotMob/Sofascore work-rate stats have enough seasons to estimate the effect; otherwise descriptive only |
| Each feeder league | provider has usable player stats for it |
| Injury history (availability model) | Transfermarkt injury pages can be scraped for the Big-5 panel |

## 3. Data layer

### Sources — one source per metric

| Source | Provides | Grain | Access |
|---|---|---|---|
| Understat | xG, xA, npxG, shots, key passes, minutes, per-match position; team xG/xGA/PPDA/deep completions; shot-level events | player-match, team-match, shot | scrape (`soccerdata`), cached |
| FotMob or Sofascore (Phase 0 picks one per metric; the other is fallback) | tackles, interceptions, recoveries, duels; feeder-league player stats; possibly possession | player-season / player-match | undocumented mobile APIs, pulled once and cached |
| Transfermarkt (dcaribou DuckDB, weekly) | market-value history, transfers with fees, DOB, detailed position, current contract end, appearances with club-at-the-time, lineups, game events | valuation-date, transfer, player-match | single file download |
| Transfermarkt injury pages (conditional) | injury spells with dates | player-spell | scrape; enters only if Phase 0 shows it is feasible for the Big-5 panel |
| ClubElo | team strength on any date | team-day | download |
| FBref (basic) | cards, appearance cross-checks; the committed 2024-25 Opta snapshot as a validation set for derived metrics | player-season | existing files; site blocks non-browser clients |

FBref lost all Opta advanced stats in January 2026 (historical seasons included); it is not a
source for xG-grade metrics. The repo's 2024-25 CSVs are irreplaceable; they move to
`legacy/data/` and stay committed (the git-ignore on `data/` does not cover `legacy/`).

Two things the DuckDB does *not* have: an injury table, and contract history (contract end is a
current field only). Neither can be used at backtest freeze dates.

### Identity

An identity table maps our `player_id` / `team_id` to every provider's ID. Seeded from the reep
register (Transfermarkt, FBref, Understat, Sofascore IDs; verified on DOB + name) and the
worldfootballR FBref↔Transfermarkt file. Leftovers matched on name + DOB + club-season; ambiguous
cases reviewed by hand and stored in a committed override file, never re-guessed at runtime.
Teams mapped to one lineage across renames and promotions. Matches join on (date, home, away).

### Canonical tables (saved to `data/processed/`)

1. `player_match` — minutes, role code, xG/xA/shots, club at the time, opponent, opponent Elo
2. `player_season` — aggregated from (1) plus work-rate stats, age at 1 July, market value at
   season start and end, injury days (conditional on the injury scrape)
3. `team_match` / `team_season` — xG, xGA, PPDA, deep completions, Elo, lineup shape → style vector
4. `market` — valuation time series, transfers with fees, contract end (current only)

Modelling matrices are derived from these in `src/`, with tests. No wide "everything" table.

### Rules

- Never mix one metric across providers (a player's xG is Understat's, always).
- Understat is the authority for minutes; other providers' minutes serve only their own per-90s.
- A mid-season move is two club-stints, not one blended row.
- Age is computed at 1 July from DOB.
- Missing is `NaN`, never 0.
- Raw fetches are never committed; `data/` is git-ignored except `data/README.md`.

## 4. Models

Each component has one job and a stated check (§5.3). *Open* choices are made on the data at a
checkpoint; rejected options and reasons are recorded in the CLAUDE.md state log.

**A. Contribution (quality).** Per player-season, per role: expected contribution per 90 with an
interval, built on expected quantities (npxG, xA, chance creation), not on goals. Built in: a
continuous opponent-strength adjustment (a per-player slope against opponent Elo, shrunk toward
the role mean), recency weighting (exponential decay over the last three seasons), and the
finishing residual (goals − xG) tracked separately. Attacking roles are measured on npxG, xA and
chance creation; midfield and defensive roles on possession-chain involvement (Understat's
`xGChain` / `xGBuildup`), work-rate stats where available, and the plus-minus variant, which is
the only measure here that captures defensive value (through xGA). So plus-minus is *required*
for non-attacking roles. *Open, for attacking roles:* team-share adjustment vs plus-minus vs both.

**B. League conversion.** Factors per league pair from movers, age-controlled, with intervals.

**C. Replacement level and surplus.** Per role, the contribution of a freely available player;
value is contribution above that level. *Open:* operationalised as the median contribution of
players signed for free or for a fee below a low percentile, or as a low percentile of all
players with ≥900 minutes in the role — decided by which is more stable across seasons.

**D. Market model.** Expected market value from stats, age, role, league, club tier. Contract
remaining is excluded: it exists only as a current field and would be unavailable at backtest
freeze dates. Residuals are the price gaps; applied to projections it yields expected resale.
*Open:* hierarchical regression vs gradient boosting with monotone constraints, judged on
held-out seasons.

**E. Trajectory.** Hierarchical aging curves by role with player-level effects; 1-3 season
projections with intervals.

**F. Availability.** Expected minutes from age, role and prior-seasons minutes; injury history
is added only if the conditional injury scrape enters. Converts per-90 into per-season
contribution.

**G. Fit.** Role fit: minutes in the requested slot plus the estimated cost of role switches
(from players who switched). Style fit: player profile against the club's style vector, effect
estimated on a matched-transfer design (similar players, different destination styles) to address
the selection problem that clubs choose whom they buy.

**H. Similarity.** A learned embedding from per-90 profile, shot locations and role minutes;
distance replaces the old cluster threshold. Feeds "players like X".

**I. Objective.** Surplus contribution per euro of cost, plus expected resale, over the horizon.
Cost is the transfer fee, or the market value at the time when the fee is undisclosed or the move
is free; wages are not publicly available and are not modelled.
Candidates ranked by the probability of matching X's contribution (or a conservative quantile),
never by the mean alone. Calibration of all shown intervals is reported.

## 5. Validation

### 5.1 Departure backtest (headline)

- *Cases:* each season 2015-16 → 2023-24, every instance of a smaller Big-5 club (bottom ~two
  thirds by squad market value; threshold chosen on the data) selling a player with ≥1,500 minutes
  for them to a club with higher squad value.
- *Procedure:* freeze all data at the sale date; refit league factors, replacement levels, aging
  curves and the market model on pre-freeze data; produce the top-5 shortlist within the fee
  received. Over the following two seasons compare against the club's actual replacement(s).
- *Outcomes:* primary outcomes are model-independent so the model is not graded by its own
  ruler — minutes share at the new club, raw goals+assists per 90, and realised market-value
  change, each per euro of cost. Our contribution metric is reported as a secondary outcome.
- *Candidate pools:* results reported for two pools — all eligible players, and only players who
  actually transferred in that window (the verifiable "could have been bought" set).
- *Baselines:* (1) the club's actual signing; (2) best raw goals+assists per 90 under budget;
  (3) highest market value under budget.
- *Case studies:* five to eight real departures written up in full, including ones the club won.

### 5.2 Leakage

No feature uses data after the freeze date; Transfermarkt values are the last update before it.
A unit test feeds a post-freeze season and asserts the pipeline refuses it.

### 5.3 Kill checks

| Component | Check | If it fails |
|---|---|---|
| Contribution | year-to-year stability > raw G+A; correlates with team xG difference; chosen variant best predicts post-move output | simpler variant |
| Opponent slope | persistent; improves post-move prediction | dropped; reported as "scout intuition unsupported" |
| Finishing residual | persistence | expected to fail; that is the finding |
| League factors | reproduce out-of-sample movers within intervals | widen / merge leagues |
| Goalkeeper proxy | stability | keepers cut, evidence shown |
| Style fit | matched-transfer effect distinguishable from zero | descriptive only, not in ranking |
| Role-switch cost | estimate per switch type | zero-cost switches treated as free |
| Availability | beats "last season's minutes" on held-out seasons | use the baseline |
| Market model | held-out-season error; monotonicity sanity | simpler family |
| Calibration | backtest coverage vs nominal for every shown interval | widen intervals, report |

### 5.4 Reproducibility and determinism

`make train` from a clean clone and in Docker reproduces every committed artifact; a test compares
regenerated key numbers to committed ones. Seeds are set everywhere; thread counts are pinned for
any fit where multithreading breaks bit-reproducibility (MCMC, boosted trees), and the writeup
states which numbers are reproducible to the bit and which to a tolerance.

Before any fit with entity effects (player effects in plus-minus, aging curves, league factors),
a check asserts that every entity in the holdout has training rows *and* that the graph
connecting entities (players via shared matches, leagues via movers) is connected; an entity
with zero rows is absent, not merely unseen.

## 6. Structure and app

Same repository; `feature/*` branches into `main`; old work under `legacy/`.

Tooling: `uv` (+ `uv.lock`, `.python-version`), `ruff` (line length 100), `pytest` mirroring
`src/`, one `Dockerfile`, `Makefile` (`setup / lint / test / spike / data / train / app / clean`),
pre-commit (ruff; notebook outputs are never stripped), CI (`uv sync --locked → ruff → pytest`).

Notebooks explore; the package ships. Exploration and each phase's checkpoint analysis happen
in numbered notebooks kept with outputs; settled logic is ported to `src/scout/` and the port is
verified to reproduce the notebook's exact numbers (row counts, diagnostics, metrics) locally and
in Docker. Diagnostics run after fits and never raise.

```
.github/workflows/ci.yml
notebooks/                numbered, kept with outputs; one per phase checkpoint
src/scout/
  data/        one module per provider — `get_*` / `load_*` return raw, never write
  identity.py  ID resolution + committed overrides
  panel/       builders for the canonical tables → data/processed/
  models/      contribution, league_factors, market, trajectory, availability, fit, similarity
  backtest.py  departure cases, freeze logic, baselines
  plots.py     Agg backend
  __main__.py  python -m scout runs the whole pipeline
models/                   committed artifacts (parquet/JSON)
reports/figures/          committed
app/                      Streamlit; reads models/ only; one file per view
tests/                    mirrors src/
data/README.md            the only committed file under data/
.env.example              documents any provider keys (none expected; kept for the template)
legacy/                   the original notebooks and CSVs, untouched
docs/superpowers/specs/   this document
```

**App contract.** `app/` imports nothing from `src/scout/models`; it reads committed artifacts
and renders them. A CI check asserts `app/` contains no hardcoded result values: every displayed
number comes from an artifact, and the only numeric literals allowed are layout constants
(sizes, limits) listed in one place. A missing artifact makes a view say so, not compute.

Views: **Replace X** (front door: club + departing player + budget → shortlist scored on
quality, price, role fit, style fit, trajectory, with reasons and intervals), **Find**
(club + slot + budget → shortlist), **Player** (trajectory vs role curve, value history with
fair-value band, output vs opponent strength, role minutes, projection, neighbours), **Club**
(style vector, minutes by role, weakest slots), **Market** (what the market pays for, what
contributes, the gap, league factors, backtest results per season).

## 7. Phases

Each phase ends in a verified checkpoint and gets its own implementation plan when reached.

0. **Spike** — throwaway code. Answers: Understat↔Transfermarkt match rates for the Big 5;
   FotMob and Sofascore player-stat depth (seasons, metrics) for the Big 5 and the feeder list;
   feeder-league ranking from Transfermarkt transfers; goalkeeper proxy feasibility; injury
   scrape feasibility. Output: `docs/superpowers/specs/<date>-phase0-findings.md`, which
   resolves every conditional item in §2 and §3.
1. **Data layer** — identity table, canonical tables, tests for leakage and `NaN` rules.
   Checkpoint: match-rate (target >95% of Big-5 player-minutes resolved) and minutes
   reconciliation report.
2. **Contribution, league factors, replacement level.** Checkpoint: kill checks, variant decided.
3. **Market, trajectory, availability.** Checkpoint: held-out errors, calibration.
4. **Fit and similarity.** Checkpoint: matched-transfer effect, role-switch costs.
5. **Departure backtest.** Checkpoint: per-season results vs three baselines; case studies chosen.
6. **App and writeup.** Checkpoint: rendered app checked view by view; every README number
   traced to an artifact.

## 8. Risks

- Provider access (FotMob/Sofascore) can change or block: pull early, cache, record the snapshot.
- Feeder-league data may be thin: tier 2 shrinks to what is verified; the writeup says so.
- Entity resolution below target: unresolved players are listed, and the effect on coverage is
  quantified before modelling starts.
- Backtest sample per season may be small for some roles: results are reported pooled and by
  role, with intervals, rather than hidden.