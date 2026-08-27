from common import BIG5, tm_connect, write_json

con = tm_connect()
comps = ",".join(f"'{c}'" for c in BIG5)
sql = f"""
WITH ranked AS (
  SELECT TRY_CAST(club_id AS INTEGER) AS club_id, domestic_competition_id,
         ROW_NUMBER() OVER (
           PARTITION BY domestic_competition_id ORDER BY total_market_value DESC NULLS LAST
         ) AS rk
  FROM clubs WHERE domestic_competition_id IN ({comps})
),
smaller AS (SELECT club_id FROM ranked WHERE rk > 6),
moves AS (
  SELECT t.player_id, t.transfer_season, TRY_CAST(t.transfer_fee AS DOUBLE) AS transfer_fee,
         t.from_club_id, t.from_club_name,
         fc.domestic_competition_id AS from_comp
  FROM transfers t
  LEFT JOIN clubs fc ON t.from_club_id = TRY_CAST(fc.club_id AS INTEGER)
  WHERE t.to_club_id IN (SELECT club_id FROM smaller)
    AND TRY_CAST(SUBSTR(t.transfer_season, 1, 2) AS INTEGER) BETWEEN 14 AND 25
)
SELECT COALESCE(m.from_comp, 'UNKNOWN') AS competition_id,
       COALESCE(c.name, 'club not in dataset') AS competition_name,
       COALESCE(c.country_name, '') AS country,
       COUNT(*) AS transfers,
       SUM(CASE WHEN m.transfer_fee > 0 THEN 1 ELSE 0 END) AS paid_transfers,
       COUNT(DISTINCT m.player_id) AS players
FROM moves m LEFT JOIN competitions c ON m.from_comp = c.competition_id
GROUP BY ALL ORDER BY transfers DESC, competition_id
"""
ranking = con.execute(sql).df()

# The clubs table only covers competitions in the dataset, so sources outside it (second tiers,
# reserve teams, free agents) show up as UNKNOWN; classify them by name so the note can say so.
unknown_sql = f"""
WITH ranked AS (
  SELECT TRY_CAST(club_id AS INTEGER) AS club_id,
         ROW_NUMBER() OVER (
           PARTITION BY domestic_competition_id ORDER BY total_market_value DESC NULLS LAST
         ) AS rk
  FROM clubs WHERE domestic_competition_id IN ({comps})
)
SELECT t.from_club_name, COUNT(*) AS n
FROM transfers t LEFT JOIN clubs fc ON t.from_club_id = TRY_CAST(fc.club_id AS INTEGER)
WHERE t.to_club_id IN (SELECT club_id FROM ranked WHERE rk > 6)
  AND TRY_CAST(SUBSTR(t.transfer_season, 1, 2) AS INTEGER) BETWEEN 14 AND 25
  AND fc.club_id IS NULL
GROUP BY 1 ORDER BY 2 DESC, 1
"""
unknown = con.execute(unknown_sql).df()
reserve = unknown.from_club_name.str.contains(
    r"(?:\b(?:B|II|U\d\d|U-?\d\d|Castilla|Next Gen|Atl\.|Youth|Reserves?|Amateure|Jong|Atlètic)\b)",
    regex=True,
)
free_agent = unknown.from_club_name.str.contains("Without Club|Retired|Unknown", regex=True)
unknown_breakdown = {
    "reserve_or_youth_teams": int(unknown.loc[reserve, "n"].sum()),
    "free_agents": int(unknown.loc[free_agent, "n"].sum()),
    "other_clubs_not_in_dataset": int(unknown.loc[~reserve & ~free_agent, "n"].sum()),
    "other_clubs_top30": unknown[~reserve & ~free_agent].head(30).to_dict("records"),
}
unknown_share = float(
    ranking.loc[ranking.competition_id == "UNKNOWN", "transfers"].sum() / ranking.transfers.sum()
)
feeders = ranking[~ranking.competition_id.isin(list(BIG5) + ["UNKNOWN"])]
write_json(
    "feeder_leagues.json",
    {
        "definition": (
            "signings by Big-5 clubs ranked 7th+ by current squad value, seasons 14/15 to 25/26"
        ),
        "unknown_source_share": unknown_share,
        "unknown_breakdown": unknown_breakdown,
        "big5_internal": ranking[ranking.competition_id.isin(BIG5)].to_dict("records"),
        "feeders_ranked": feeders.to_dict("records"),
        "FEEDER_TOP8": feeders.head(8).competition_id.tolist(),
    },
)
print(f"unknown-source share: {unknown_share:.1%}")
print(feeders.head(15).to_string(index=False))
