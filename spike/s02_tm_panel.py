from common import BIG5, SEASONS, SPIKE, tm_connect, write_json

out = SPIKE / "tm_pcs.parquet"
con = tm_connect()
comps = ",".join(f"'{c}'" for c in BIG5)
sql = f"""
SELECT a.player_id, p.name, p.date_of_birth, p.position, p.sub_position,
       a.player_club_id AS club_id, c.name AS club_name, g.competition_id,
       CAST(g.season AS INTEGER) AS season,
       SUM(a.minutes_played) AS minutes, COUNT(*) AS apps
FROM appearances a
JOIN games g ON a.game_id = g.game_id
JOIN players p ON a.player_id = p.player_id
JOIN clubs c ON a.player_club_id = c.club_id
WHERE g.competition_id IN ({comps})
  AND CAST(g.season AS INTEGER) BETWEEN {SEASONS[0]} AND {SEASONS[-1]}
GROUP BY ALL
"""
panel = con.execute(sql).df()
panel.to_parquet(out, index=False)

summary = {
    "rows": len(panel),
    "players": int(panel.player_id.nunique()),
    "clubs": int(panel.club_id.nunique()),
    "by_season": panel.groupby("season")
    .agg(rows=("player_id", "size"), minutes=("minutes", "sum"))
    .reset_index()
    .to_dict("records"),
    "by_competition": panel.groupby("competition_id").size().to_dict(),
}
write_json("tm_panel_summary.json", summary)
