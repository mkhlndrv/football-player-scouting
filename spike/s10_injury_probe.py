from io import StringIO

import pandas as pd
from common import SPIKE, polite_get, tm_connect, write_json

tm = pd.read_parquet(SPIKE / "tm_pcs.parquet")
top = tm.groupby("player_id").minutes.sum().nlargest(8).index.tolist()
con = tm_connect()
ids = ",".join(str(i) for i in top)
players = con.execute(f"SELECT player_id, name, url FROM players WHERE player_id IN ({ids})").df()

records = []
for _, p in players.iterrows():
    url = str(p.url).replace("/profil/", "/verletzungen/")
    r = polite_get(url)
    rec = {"player": p["name"], "url": url, "status": r.status_code}
    if r.status_code == 200:
        try:
            tables = pd.read_html(StringIO(r.text))
        except ValueError:
            tables = []
        hit = next((t for t in tables if any("injury" in str(c).lower() for c in t.columns)), None)
        if hit is not None:
            rec["columns"] = [str(c) for c in hit.columns]
            rec["rows"] = len(hit)
            season_col = next((c for c in hit.columns if "season" in str(c).lower()), None)
            rec["earliest_season"] = str(hit[season_col].astype(str).min()) if season_col else None
            rec["sample_rows"] = hit.head(3).astype(str).to_dict("records")
    else:
        rec["body_head"] = r.text[:150]
    records.append(rec)

write_json(
    "injury_probe.json",
    {
        "probed": len(records),
        "ok": sum(r["status"] == 200 for r in records),
        "parsed": sum("rows" in r for r in records),
        "records": records,
    },
)
