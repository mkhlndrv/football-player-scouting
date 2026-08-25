# Football player scouting by playing-style clustering

A scouting tool that clusters 460 attacking players from Europe's big five leagues by playing
style, separately in 11 stat categories, and calls two players similar when they land in the same
cluster in most of them. Season 2024-25.

## Overview

A scout replacing a departing forward wants the same profile, not just similar headline numbers.
One clustering over all stats at once mixes styles, because a single strong dimension can pull
very different players together. This project instead clusters each stat category on its own
(shooting, passing, carrying, aerial play, and so on) and scores similarity as the number of
categories, out of 11, in which two players share a cluster.

## Data

Season aggregate statistics for 2024-25, sourced from public football statistics databases
(FBref-style tables, per `data/data_README.markdown`). 460 players whose position includes a
forward role (FW, FW/MF, MF/FW, DF/FW): 108 from La Liga, 96 Premier League, 93 Serie A,
80 Ligue 1, 73 Bundesliga, and 10 listed under two competitions after a mid-season move.
The 11 categories are goals, expected goals, shooting, passing, shot-creating actions,
goal-creating actions, carries, touches, take-ons, defence and aerials. Raw and cleaned CSVs are
committed under `data/`, so nothing needs to be scraped, and all counting stats are normalized
per 90 minutes before clustering.

## Method

**Baseline** (`notebooks/02_baseline_model2.ipynb`): one k-means over 15 per-90 features drawn
from five categories, k = 4 by the elbow method. Silhouette score 0.174, and the five nearest
neighbours it returns for Mbappé (Jérémie Boga, Robin Hack, Takefusa Kubo, Paul Wanner, Stephy
Mavididi) span three of its own four clusters. That mixing is what the main model is built to fix.

**Main model** (`model/03_main_model.ipynb`): a separate k-means per category on standardized
per-90 features, with k read off elbow and silhouette curves for each: Goals 7, xG 5, Shooting 6,
Passing 5, SCA 6, GCA 4, Carries 4, Touches 5, Take-ons 6, Defence 5, Aerial 4. The cluster
labels are combined into a 460 x 460 co-occurrence matrix, and pairs sharing a cluster in 8 or
more of the 11 categories are kept as matches: 1,195 pairs in total.

## Results

The committed outputs (`model/co_occurrence_matrix.csv` and `model/similar_player_pairs.csv`)
hold the results without rerunning anything.

- Mbappé's closest matches are Ousmane Dembélé (9 of 11 categories) and Vinícius Júnior (8).
- `find_similar_players("Serhou Guirassy")` returns Alexander Sørloth (9) and members of a tie
  at 8 that includes Erling Haaland and Jonathan David.
- Three pairs agree in all 11 categories: Calvert-Lewin and Bertuğ Yıldırım, Mateta and Dovbyk,
  Sané and Coman.

## Usage

```bash
pip install -r requirements.txt
jupyter notebook
```

Run in order: `notebooks/01_data_analysis_and_preprocessing.ipynb` (cleaning and per-90 feature
engineering), `notebooks/02_baseline_model2.ipynb` (baseline), `model/03_main_model.ipynb`
(clustering and similarity). The notebooks were written in Google Colab and read absolute
`/content/` paths, so point them at `data/` when running locally. To query similarity without
refitting, run only the "Finding Similar Players" section of the main notebook, which reads the
committed matrix.

## Limitations

- One season of data. A player's clusters reflect one year at one club, and mid-season movers
  appear as aggregated "2 Teams" rows.
- k for each category was chosen by reading elbow and silhouette plots. The main model's cluster
  quality is not quantified, and the baseline's silhouette of 0.174 is weak.
- All 11 categories count equally in the similarity score. A scout would weight shooting and
  passing differently for a striker and a winger.
- No validation against outcomes: the matches are face-plausible, but nothing tests them against
  real transfers or later performance.

## Layout

```
notebooks/
  01_data_analysis_and_preprocessing.ipynb   clean the raw tables, engineer per-90 features
  02_baseline_model2.ipynb                   baseline: one k-means over all features
model/
  03_main_model.ipynb                        per-category k-means and co-occurrence similarity
  co_occurrence_matrix.csv                   460 x 460 shared-category counts (committed)
  similar_player_pairs.csv                   1,195 pairs sharing 8+ of 11 categories (committed)
data/
  original/                                  raw season tables
  cleaned/                                   cleaned per-category tables
  player_data_for_clustering.csv             merged baseline dataset
requirements.txt
```
