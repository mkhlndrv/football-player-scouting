import json
import re

import pandas as pd
from common import OUT, SPIKE, polite_get, tm_connect, write_json

SEARCH_URL = "https://apigw.fotmob.com/searchapi/suggest"  # www.fotmob.com/api/searchapi is 404
STATS_URL = (
    "https://www.fotmob.com/api/data/playerStats"  # /api/playerStats and /api/playerData are 404
)
PAGE_URL = "https://www.fotmob.com/players/{pid}/x"
NEXT_DATA = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
TITLE = re.compile(r'"title"\s*:\s*"([^"]+)"')


def sample_players():
    tm = pd.read_parquet(SPIKE / "tm_pcs.parquet")
    per = tm.groupby(["competition_id", "player_id", "name"]).season.nunique().reset_index()
    big5 = per.sort_values("season", ascending=False).groupby("competition_id").head(2)
    feeders = json.loads((OUT / "feeder_leagues.json").read_text())["FEEDER_TOP8"]
    con = tm_connect()
    comps = ",".join(f"'{c}'" for c in feeders)
    extra = con.execute(
        f"""SELECT competition_id, player_id, name, seasons FROM (
                SELECT c.domestic_competition_id AS competition_id, p.player_id, p.name,
                       COUNT(DISTINCT g.season) AS seasons
                FROM appearances a
                JOIN games g ON a.game_id = g.game_id
                JOIN clubs c ON a.player_club_id = c.club_id
                JOIN players p ON a.player_id = p.player_id
                WHERE c.domestic_competition_id IN ({comps})
                  AND CAST(g.season AS INTEGER) >= 2014
                GROUP BY 1, 2, 3
            ) QUALIFY ROW_NUMBER() OVER (PARTITION BY competition_id ORDER BY seasons DESC) = 1"""
    ).df()
    return pd.concat([big5, extra])[["competition_id", "name"]].to_dict("records")


def season_stats(pid, entry_id):
    r = polite_get(STATS_URL, params={"playerId": pid, "seasonId": entry_id})
    return r.status_code, sorted(set(TITLE.findall(r.text))) if r.status_code == 200 else []


def probe_player(name):
    rec = {"query": name}
    r = polite_get(SEARCH_URL, params={"term": name, "lang": "en"})
    rec["search_status"] = r.status_code
    ids = re.findall(r'"id"\s*:\s*"?(\d{4,8})"?', r.text) if r.status_code == 200 else []
    if not ids:
        return rec
    pid = rec["fotmob_id"] = ids[0]
    page = polite_get(PAGE_URL.format(pid=pid))
    rec["page_status"] = page.status_code
    m = NEXT_DATA.search(page.text)
    if not m:
        return rec
    data = json.loads(m.group(1))["props"]["pageProps"]["data"]
    seasons = data.get("statSeasons", [])
    rec["seasons"] = [s["seasonName"] for s in seasons]
    deep = [s for s in seasons if any(t.get("hasDeepStats") for t in s["tournaments"])]
    rec["deep_seasons"] = [s["seasonName"] for s in deep]
    rec["earliest_deep_season"] = min(rec["deep_seasons"], default=None)
    rec["contract_end"] = data.get("contractEnd")
    rec["has_market_values"] = bool(data.get("marketValues"))
    if deep:
        earliest = min(deep, key=lambda s: s["seasonName"])
        latest = deep[0]
        for label, s in (("earliest_deep", earliest), ("latest", latest)):
            status, titles = season_stats(pid, s["tournaments"][0]["entryId"])
            rec[f"{label}_season"] = s["seasonName"]
            rec[f"{label}_status"] = status
            rec[f"{label}_titles"] = titles
        # The flag can be set while the season returns almost nothing; walk forward to the
        # first season whose domestic-league entry actually returns a full stat set.
        rec["first_full_deep_season"] = None
        for s in sorted(deep, key=lambda s: s["seasonName"])[:7]:
            status, titles = season_stats(pid, s["tournaments"][0]["entryId"])
            if status == 200 and len(titles) >= 20:
                rec["first_full_deep_season"] = s["seasonName"]
                rec["first_full_deep_n_titles"] = len(titles)
                break
    return rec


results = [
    probe_player(p["name"]) | {"competition_id": p["competition_id"]} for p in sample_players()
]
all_titles = sorted({t for r in results for t in r.get("latest_titles", [])})
summary = {
    "endpoints": {
        "search": SEARCH_URL,
        "player_page_next_data": PAGE_URL,
        "season_stats": STATS_URL + "?playerId=<id>&seasonId=<entryId>",
        "dead": ["www.fotmob.com/api/playerData", "www.fotmob.com/api/playerStats"],
    },
    "players_probed": len(results),
    "found_ids": sum("fotmob_id" in r for r in results),
    "with_seasons": sum("seasons" in r for r in results),
    "earliest_deep_season_by_player": {
        f"{r['competition_id']}:{r['query']}": r.get("earliest_deep_season") for r in results
    },
    "first_full_deep_season_by_player": {
        f"{r['competition_id']}:{r['query']}": r.get("first_full_deep_season") for r in results
    },
    "deep_stat_titles_union": all_titles,
    "records": results,
}
write_json("fotmob_probe.json", summary)
print("first full deep season:", summary["first_full_deep_season_by_player"])
print(f"{len(all_titles)} deep stat titles")
