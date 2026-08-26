import pandas as pd
from common import BIG5, OUT, ROOT, SPIKE, write_json
from rapidfuzz import fuzz, process

from scout.identity import normalize_name

us = pd.read_parquet(SPIKE / "us_ps.parquet")
tm = pd.read_parquet(SPIKE / "tm_pcs.parquet")
overrides = pd.read_csv(ROOT / "spike" / "overrides" / "team_map_overrides.csv")

rows = []
for comp, league in BIG5.items():
    clubs = tm[tm.competition_id == comp][["club_id", "club_name"]].drop_duplicates()
    choices = {
        normalize_name(n): (cid, n) for cid, n in zip(clubs.club_id, clubs.club_name, strict=True)
    }
    names_by_id = clubs.set_index("club_id").club_name
    for team in sorted(us[us.league == league].team.unique()):
        ov = overrides[(overrides.league == league) & (overrides.understat_team == team)]
        if not ov.empty:
            cid = int(ov.club_id.iloc[0])
            rows.append((league, team, cid, names_by_id.get(cid), 100, "override"))
            continue
        hits = process.extract(
            normalize_name(team), list(choices), scorer=fuzz.token_set_ratio, limit=5
        )
        score = hits[0][1]
        # token_set_ratio scores any token subset as 100 ("Barcelona" matches both
        # "FC Barcelona" and "RCD Espanyol Barcelona"); break ties by the shortest name.
        norm = min((h[0] for h in hits if h[1] == score), key=len)
        cid, name = choices[norm]
        ok = score >= 90
        rows.append((league, team, cid if ok else None, name if ok else None, score, "auto"))

team_map = pd.DataFrame(
    rows, columns=["league", "understat_team", "club_id", "club_name", "score", "source"]
)
team_map.to_csv(OUT / "team_map.csv", index=False)
unresolved = team_map[team_map.club_id.isna()]
unresolved.to_csv(OUT / "team_map_unresolved.csv", index=False)
dupes = team_map.dropna().groupby(["league", "club_id"]).size()
write_json(
    "team_map_summary.json",
    {
        "teams": len(team_map),
        "resolved": int(team_map.club_id.notna().sum()),
        "unresolved": len(unresolved),
        "duplicate_targets": int((dupes > 1).sum()),
    },
)
