import pandas as pd
from common import BIG5, OUT, SPIKE, tm_connect, write_json

from scout.identity import match_players

us = pd.read_parquet(SPIKE / "us_ps.parquet")
tm = pd.read_parquet(SPIKE / "tm_pcs.parquet")
team_map = pd.read_csv(OUT / "team_map.csv").dropna(subset=["club_id"])

us = us.merge(
    team_map[["league", "understat_team", "club_id"]],
    left_on=["league", "team"],
    right_on=["league", "understat_team"],
    how="left",
)
left = us.rename(columns={"player": "name"})[["name", "club_id", "season", "minutes", "league"]]
left["club_key"] = left["club_id"].astype("Int64").astype(str)

tm_big5 = tm[tm.competition_id.isin(BIG5)]
right = tm_big5.rename(columns={"player_id": "right_id"})[["right_id", "name", "club_id", "season"]]

# The appearances table has holes (e.g. Atlético, Burnley, Villarreal 2014-15); game_lineups
# covers every game, so the Phase 1 identity table should be built from the union of both.
comps = ",".join(f"'{c}'" for c in BIG5)
lineups = (
    tm_connect()
    .execute(f"""
    SELECT DISTINCT l.player_id AS right_id, p.name, l.club_id, CAST(g.season AS INTEGER) AS season
    FROM game_lineups l
    JOIN games g ON l.game_id = g.game_id
    JOIN players p ON l.player_id = p.player_id
    WHERE g.competition_id IN ({comps}) AND CAST(g.season AS INTEGER) BETWEEN 2014 AND 2025
""")
    .df()
)
right_union = pd.concat([right, lineups]).drop_duplicates(["right_id", "club_id", "season"])
for frame in (right, right_union):
    frame["club_key"] = frame["club_id"].astype("Int64").astype(str)

matched_apps = match_players(left, right)
matched_apps["hit"] = matched_apps["right_id"].notna()
matched = match_players(left, right_union)
matched["hit"] = matched["right_id"].notna()


def rate(frame):
    return float(frame.loc[frame.hit, "minutes"].sum() / frame["minutes"].sum())


report = {
    "tm_side": "appearances + game_lineups",
    "appearances_only_minutes_rate": rate(matched_apps),
    "overall_minutes_rate": rate(matched),
    "overall_player_rate": float(matched.hit.mean()),
    "by_method": matched.method.value_counts().to_dict(),
    "by_league_season": [
        {"league": lg, "season": int(s), "minutes_rate": rate(g), "players": len(g)}
        for (lg, s), g in matched.groupby(["league", "season"])
    ],
    "unmatched_minutes_top20": matched[~matched.hit]
    .nlargest(20, "minutes")[["league", "season", "name", "minutes"]]
    .to_dict("records"),
}
matched[~matched.hit].sort_values("minutes", ascending=False)[
    ["league", "season", "name", "club_id", "minutes"]
].to_csv(OUT / "unmatched_players.csv", index=False)
write_json("match_rates.json", report)
print(f"minutes-weighted match rate: {report['overall_minutes_rate']:.3%}")
