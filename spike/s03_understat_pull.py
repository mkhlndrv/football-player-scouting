import pandas as pd
import soccerdata as sd
from common import BIG5, CACHE, SEASONS, SPIKE, write_json

out = SPIKE / "us_ps.parquet"
if not out.exists():
    us = sd.Understat(
        leagues=list(BIG5.values()),
        seasons=[f"{y}-{y + 1}" for y in SEASONS],
        data_dir=CACHE.parent / "understat",
    )
    ps = us.read_player_season_stats().reset_index()
    ps.columns = [str(c).lower() for c in ps.columns]
    ps["season"] = ps["season"].astype(str).str[:2].map(lambda s: 2000 + int(s))
    ps.to_parquet(out, index=False)

ps = pd.read_parquet(out)
required = {"league", "season", "team", "player", "minutes"}
missing = sorted(required - set(ps.columns))
coverage = {
    "columns": list(ps.columns),
    "missing_required": missing,
    "rows": len(ps),
    "by_league_season": ps.groupby(["league", "season"])
    .agg(players=("player", "size"), minutes=("minutes", "sum"))
    .reset_index()
    .to_dict("records"),
    "has_xg_chain": "xg_chain" in ps.columns,
    "has_xg_buildup": "xg_buildup" in ps.columns,
    "has_position": "position" in ps.columns,
}
write_json("understat_coverage.json", coverage)
assert not missing, missing
