import numpy as np
import pandas as pd
import soccerdata as sd
from common import CACHE, SPIKE, write_json

SEASONS = ["2021-2022", "2022-2023", "2023-2024"]
ON_TARGET = {"Goal", "Saved Shot"}
out = SPIKE / "gk_pl.parquet"

if not out.exists():
    us = sd.Understat(
        leagues=["ENG-Premier League"], seasons=SEASONS, data_dir=CACHE.parent / "understat"
    )
    pm = us.read_player_match_stats().reset_index()
    shots = us.read_shot_events().reset_index()
    pm[["game_id", "season", "team", "player", "position", "minutes"]].to_parquet(
        SPIKE / "pl_player_match.parquet", index=False
    )
    shots[
        ["game_id", "season", "team", "player", "xg", "result", "situation", "minute"]
    ].to_parquet(SPIKE / "pl_shots.parquet", index=False)

    keepers = (
        pm[pm["position"] == "GK"]
        .sort_values("minutes", ascending=False)
        .drop_duplicates(["game_id", "team"])[["season", "game_id", "team", "player", "minutes"]]
    )
    teams_in_game = pm.groupby("game_id")["team"].unique()
    shots["defending_team"] = [
        next((t for t in teams_in_game[g] if t != s), None)
        for g, s in zip(shots["game_id"], shots["team"], strict=True)
    ]
    on_target = shots[shots["result"].isin(ON_TARGET)]
    faced = (
        on_target.groupby(["season", "game_id", "defending_team"])
        .agg(xg_faced=("xg", "sum"), conceded=("result", lambda r: (r == "Goal").sum()))
        .reset_index()
        .rename(columns={"defending_team": "team"})
    )
    gk = keepers.merge(faced, on=["season", "game_id", "team"], how="left").fillna(
        {"xg_faced": 0, "conceded": 0}
    )
    gk["match_no"] = gk.groupby(["season", "player"]).cumcount()
    gk.to_parquet(out, index=False)

gk = pd.read_parquet(out)


def per90(frame):
    g = (
        frame.groupby(["season", "player"])
        .agg(minutes=("minutes", "sum"), xg_faced=("xg_faced", "sum"), conceded=("conceded", "sum"))
        .reset_index()
    )
    g["prevented_per90"] = (g.xg_faced - g.conceded) / g.minutes * 90
    return g


season_tab = per90(gk)
wide = season_tab[season_tab.minutes >= 1500].pivot(
    index="player", columns="season", values="prevented_per90"
)
pairs = []
cols = list(wide.columns)
for a, b in zip(cols, cols[1:], strict=False):
    both = wide[[a, b]].dropna()
    pairs.append(
        {
            "from": str(a),
            "to": str(b),
            "n": len(both),
            "pearson": float(both[a].corr(both[b])),
            "spearman": float(both[a].rank().corr(both[b].rank())),  # no scipy in the spike
        }
    )

halves = per90(
    gk.assign(season=gk.season.astype(str) + np.where(gk.match_no % 2 == 0, "_even", "_odd"))
)
halves["base"] = halves.season.str.replace(r"_(even|odd)$", "", regex=True)
halves["half"] = halves.season.str.extract(r"_(even|odd)$")[0]
hw = (
    halves[halves.minutes >= 700]
    .pivot_table(index=["base", "player"], columns="half", values="prevented_per90")
    .dropna()
)
split_half = float(hw["even"].corr(hw["odd"])) if len(hw) > 5 else None

write_json(
    "gk_proxy.json",
    {
        "scope": "ENG-Premier League 2021-22 to 2023-24, on-target shots = Goal + Saved Shot",
        "interpretation": (
            "Understat xG is pre-shot, so on-target xG faced is far below goals conceded and "
            "prevented_per90 is negative for every keeper; only the within-season ranking is "
            "meaningful. Stability is judged on that ranking (year-to-year and split-half)."
        ),
        "season_totals": season_tab.groupby("season")
        .agg(conceded=("conceded", "sum"), xg_faced=("xg_faced", "sum"))
        .reset_index()
        .to_dict("records"),
        "keeper_seasons_1500min": int((season_tab.minutes >= 1500).sum()),
        "year_to_year": pairs,
        "split_half_pearson": split_half,
        "split_half_n": int(len(hw)),
        "season_table": season_tab.sort_values(
            ["season", "prevented_per90"], ascending=[True, False]
        ).to_dict("records"),
    },
)
print("year-to-year:", pairs, "| split-half:", split_half, "n=", len(hw))
