# Phase 0 findings — 2026-08-27

Spec: `docs/superpowers/specs/2026-08-26-moneyball-replacement-scouting-design.md`, §7 Phase 0.
Every number below comes from a file in `spike/out/`; the file is named next to it. The spike
code in `spike/` is throwaway; the identity cascade it exercised lives in `src/scout/identity.py`.

## Q1. Understat ↔ Transfermarkt match rate (`match_rates.json`, `unmatched_players.csv`)

Panel: 32,574 Understat player-seasons (`understat_coverage.json`) matched against 33,061
Transfermarkt player-club-seasons (`tm_panel_summary.json`), Big 5, 2014-15 → 2025-26, through a
team map of 168 Understat names → Transfermarkt club ids (`team_map_summary.json`: 168/168
resolved, 11 hand overrides, one legitimate duplicate — Understat renamed Parma mid-period).

- **99.55% of player-minutes and 99.44% of player-seasons resolve** to a Transfermarkt id.
  Lowest league-season: La Liga 2017-18 at 98.5%.
- Method split: exact 30,185 · surname 1,407 · fuzzy ≥92 751 · unique-in-season 47 ·
  unmatched 184 (190,476 minutes; 85 player-seasons with ≥900 min).
- The Transfermarkt `appearances` table has holes: Atlético, Burnley and Villarreal 2014-15 hold
  3–4 players each, with partial gaps for Barcelona, Real Madrid and Valencia 2020-21. With
  appearances alone the rate was 98.96%; building the Transfermarkt side from
  `appearances ∪ game_lineups` (lineups cover every game) lifts it to 99.55%. This also explains
  the 2014-15 minute total being 4.5% below 2015-16 in `tm_panel_summary.json`.
- What remains: 35 mononyms/nicknames with no token overlap ("Bono" for Yassine Bounou, "Mario",
  "Sávio") and name variants ("Franck Zambo", "Kelvin Adou"). No cascade solves these; they need
  an alias table.
- A "token subset" cascade stage was tried and reverted: it changed zero outcomes.

**Decision.** The Phase 1 target (>95% of Big-5 player-minutes) is met without hand overrides at
player level. Phase 1 builds the Transfermarkt player-club-season table from
`appearances ∪ game_lineups` and seeds aliases from the reep register for the mononym leftovers.
Rejected: hand-patching individual players in the spike — unmaintainable and hides the cause.

## Q2. Work-rate stats depth (`fotmob_probe.json`, `sofascore_probe.json`)

Sample: 26 players — the two longest-tenured per Big-5 league and per top-8 feeder league,
chosen deterministically (seasons in the panel, then player id), so the probe reproduces.

**FotMob** — 26/26 found. The documented JSON endpoints (`/api/playerData`, `/api/playerStats`)
are dead (404); the search API at `apigw.fotmob.com`, the player page's embedded JSON, and
`/api/data/playerStats?playerId&seasonId=<entryId>` all answer 200 without a token. 72 distinct
per-season stat titles (outfield and keeper), including tackles, interceptions, recoveries,
duels won, aerials won, dribbles, dispossessed, touches (incl. opposition box), possession won
final third, pass accuracy, chances created, xG, xA, xGOT, and two on/off-pitch numbers ("xG
against while on pitch", "goals conceded while on pitch").
First season returning a full stat set, taking the
earlier of the two players per league: **2016-17 for all ten Big-5 players**; Brazil 2016-17;
Netherlands, Portugal and Switzerland 2017-18; Denmark 2017 (calendar-year season); Belgium,
Austria and Turkey 2018-19. 2014-15 and 2015-16 carry only goals, assists, minutes and cards.

**Sofascore** — 26/26 found. Plain HTTP is refused (403) on all three hosts by TLS
fingerprinting; the browser-fingerprint client `soccerdata` already depends on (`tls_requests`)
answers 200 everywhere, with no throttling across ~150 requests at 3 s spacing. 110–117 keys per
player-season: everything FotMob has plus ground vs aerial duels, clearances, blocked shots,
dribbled past, possession lost, errors leading to shot/goal, passes by zone (own half, opposition
half, final third), crosses, long balls, big chances created/missed — and a complete goalkeeper
block (saves caught/parried, inside/outside the box, high claims, punches, penalties faced/saved,
goals conceded by zone). First full season, earlier of the two players per league: **2015-16 for
all ten Big-5 players**; Netherlands, Turkey, Brazil and Switzerland 2015-16; Portugal 2016-17;
Austria 2017-18; Denmark 2019-20; Belgium 2020-21. For Belgium and Denmark both sampled players
agree, and FotMob reaches further back there (2018-19 and 2017), so the deeper provider is
league-specific.

**Decision.** Sofascore is the primary provider for work-rate metrics and keeper stats in the
Big 5 and in Netherlands, Turkey, Brazil, Switzerland, Portugal and Austria (deeper, richer, has
keepers); FotMob is primary for Belgium and Denmark and the fallback elsewhere, and the only
source of the on/off-pitch numbers. Work-rate features exist for 2015-16 → 2025-26 in the Big 5
(11 seasons), so **style fit enters the ranking** with a backtestable history; its transfer test
runs 2016-17 → 2023-24. The spec's guess that FotMob would be primary is reversed for most
leagues. Caveat: 26 players is not a census — Phase 1 measures coverage across the whole panel.

## Q3. Feeder leagues (`feeder_leagues.json`)

Definition: signings by Big-5 clubs ranked 7th or lower in their league by *current* squad market
value (historical squad value is not in the DuckDB), seasons 14/15 → 25/26; loans and returns are
included because the transfers table does not flag them.

| # | League | Signings | Paid |
|---|---|---|---|
| 1 | Belgium | 515 | 189 |
| 2 | Netherlands | 461 | 184 |
| 3 | Portugal | 424 | 153 |
| 4 | Turkey | 303 | 56 |
| 5 | Switzerland | 198 | 83 |
| 6 | Brazil | 185 | 101 |
| 7 | Austria | 177 | 77 |
| 8 | Denmark | 166 | 91 |
| 9–15 | Scotland 160 · Argentina 134 · Greece 131 · MLS 103 · Poland 88 · Croatia 87 · Sweden 87 | | |

Moves between Big-5 clubs dominate (Serie A 2,814, PL 2,049, La Liga 1,672, Ligue 1 1,601,
Bundesliga 1,360). 27.8% of signings come from clubs with no league in the dataset: 2,343 from
reserve and youth teams (internal promotions, not scouting), 121 free agents, and **2,613 from
clubs outside the dataset — overwhelmingly the Big-5 second tiers** (Birmingham, Derby,
Blackburn, Bolton, Preston, Coventry; Perugia, Bari, Ascoli; Mirandés; Kaiserslautern). The
Championship and its peers are missing from the ranking because the DuckDB has no clubs table for
them, not because they are small feeders.

**Decision.** Tier 2 = Belgium, Netherlands, Portugal, Turkey, Switzerland, Brazil, Austria,
Denmark. By Q2 depth all eight have ≥5 work-rate seasons on their deeper provider (the shallowest
is Belgium, 2018-19 onward via FotMob = 8 seasons), so all are eligible for conversion-factor
estimation, re-checked on the full panel in Phase 1. The Big-5 second tiers are an explicit
Phase 1 coverage question: their transfers are in the data by club name; only the league lookup
is absent. Rejected: ranking by paid transfers
only — it would drop Turkey (56 paid of 303), which is a real source of free and loan moves.

## Q4. Goalkeeper proxy (`gk_proxy.json`)

Premier League 2021-22 → 2023-24: 1,140 matches, 2,280 keeper-match rows, 64 keeper-seasons with
≥1,500 minutes. Proxy = (xG of on-target shots faced − goals conceded) per 90. Season totals
reconcile with the league (goals conceded 1,037 / 1,038 / 1,196, all 380 games per season).

- Year-to-year correlation of the ranking: **r = 0.52** (2021-22 → 2022-23, n = 14) and
  **r = 0.42** (2022-23 → 2023-24, n = 9); Spearman 0.47 / 0.50.
- Split-half within season (odd vs even matches, ≥700 min per half): **r = 0.30** (n = 66).
- Face validity: Alisson, Ederson, Leno, Pope, Emiliano Martínez, Raya at the top; Meslier,
  Krul, Foster, Bazunu, Trafford at the bottom.
- The absolute level is negative for every keeper (−0.2 to −1.0) because Understat's xG is
  pre-shot and on-target shots carry far less xG than the goals they produce; only the ranking is
  meaningful, and it does not separate keeper from defence the way post-shot xG would.

**Decision.** Against the plan's bar (year-to-year ≳ 0.3, split-half ≳ 0.5) the proxy passes the
first and misses the second: a real, persistent, noisy signal. **Goalkeepers stay in**, modelled
with shrinkage toward the mean, and combined in Phase 2 with Sofascore's independent keeper block
(saves by zone, high claims, goals conceded by zone). Rejected: cutting keepers — the signal is
comparable to what a single season of outfield per-90 rates shows; and ranking keepers on one
unshrunk season — the split-half number says that would be mostly noise.

## Q5. Injury pages (`injury_probe.json`)

8/8 Transfermarkt injury pages answered 200 to a plain client; 8/8 parsed with `read_html` into
`Season, Injury, from, until, Days, Games missed`, 6–15 spells per player, histories back to
2009-10. The URL derives from the DuckDB's `players.url` (`/profil/` → `/verletzungen/`), so there
is no identity work. Full Big-5 panel ≈ 9,400 players × 3 s ≈ 8 hours once, cached.

**Decision.** The injury scrape enters; the availability model uses injury days and games missed
alongside minutes history and age.

## Understat coverage (`understat_coverage.json`)

All players, all positions, 12 seasons, 32,574 player-seasons; `xg_chain`, `xg_buildup` and
season position present, so the contribution model's midfield/defender basis in spec §4.A is
available. Minutes per league-season reconcile with fixtures (0.75M for 380-game leagues, 0.60M
Bundesliga, 0.55M for the COVID-cut Ligue 1 2019-20). Per-match pages carry the role codes the
spec's taxonomy needs (`DC, DL, DR, DMC, MC, ML, MR, AMC, AML, AMR, FW, GK`), verified on the 1,140
PL matches fetched for Q4. Understat drops the connection roughly every 120 requests; Phase 1
fetchers must retry and resume from cache.

## What changes in the spec

- §2 Conditional items — all resolved: goalkeepers **in** (shrinkage, two metrics); style fit
  **in the ranking**; feeder leagues = the eight above; injury history **in**.
- §3 Sources — Sofascore primary for work-rate and keeper stats via a TLS-fingerprint client in
  the Big 5 and most feeders, FotMob primary for Belgium and Denmark and fallback elsewhere (and
  the on/off-pitch numbers); Transfermarkt player-club-season built from
  `appearances ∪ game_lineups`; reep aliases for mononyms; Transfermarkt injury pages in.
- §2 Tier 1 — work-rate features start 2015-16, xG-grade features 2014-15; the style-fit
  transfer test window is 2016-17 → 2023-24.
- §8 Risks — add: Big-5 second tiers absent from the feeder ranking (coverage gap in the
  Transfermarkt dataset, not a finding about them); Transfermarkt appearances holes closed by
  lineups.
