import json

import pandas as pd
from common import OUT, SPIKE, polite_get, tm_connect, write_json

API = "https://api.sofascore.com/api/v1"
SS_HEADERS = {"Referer": "https://www.sofascore.com/", "Origin": "https://www.sofascore.com"}
NON_DOMESTIC = (
    "UEFA",
    "World Cup",
    "European Championship",
    "Nations League",
    "Copa",
    "U21",
    "U20",
    "U19",
)


def sample_players():
    tm = pd.read_parquet(SPIKE / "tm_pcs.parquet")
    per = tm.groupby(["competition_id", "player_id", "name"]).season.nunique().reset_index()
    big5 = (
        per.sort_values(["season", "player_id"], ascending=[False, True])
        .groupby("competition_id")
        .head(2)
    )
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
            ) QUALIFY ROW_NUMBER() OVER (
                PARTITION BY competition_id ORDER BY seasons DESC, player_id
            ) <= 2"""
    ).df()
    sample = pd.concat([big5, extra])[["competition_id", "name"]]
    return sample.sort_values(["competition_id", "name"]).to_dict("records")


def probe_player(name):
    rec = {"query": name}
    r = polite_get(f"{API}/search/all", params={"q": name}, headers=SS_HEADERS, tls=True)
    rec["search_status"] = r.status_code
    if r.status_code != 200:
        rec["search_body"] = r.text[:200]
        return rec
    players = [x for x in r.json().get("results", []) if x.get("type") == "player"]
    if not players:
        return rec
    pid = players[0]["entity"]["id"]
    rec["sofascore_id"] = pid
    r = polite_get(f"{API}/player/{pid}/statistics/seasons", headers=SS_HEADERS, tls=True)
    rec["seasons_status"] = r.status_code
    if r.status_code != 200:
        return rec
    pairs = []
    for ut in r.json().get("uniqueTournamentSeasons", []):
        for s in ut.get("seasons", []):
            pairs.append(
                {
                    "tournament": ut["uniqueTournament"]["name"],
                    "ut_id": ut["uniqueTournament"]["id"],
                    "season": s["year"],
                    "season_id": s["id"],
                }
            )
    rec["season_pairs"] = pairs
    rec["first_full_season"] = None
    # Deep stats exist nowhere before ~2013, so start the walk at 13/14 (2-digit season strings
    # compare correctly within 2000-2099) and cap the requests per player.
    domestic = sorted(
        (
            p
            for p in pairs
            if not any(w in p["tournament"] for w in NON_DOMESTIC) and p["season"] >= "13/14"
        ),
        key=lambda p: p["season"],
    )
    for p in domestic[:10]:
        rs = polite_get(
            f"{API}/player/{pid}/unique-tournament/{p['ut_id']}/season/{p['season_id']}"
            "/statistics/overall",
            headers=SS_HEADERS,
            tls=True,
        )
        if rs.status_code == 200 and len(rs.json().get("statistics", {})) >= 60:
            rec["first_full_season"] = p
            break
    for label, p in (("earliest", pairs[-1]), ("latest", pairs[0])) if pairs else ():
        rs = polite_get(
            f"{API}/player/{pid}/unique-tournament/{p['ut_id']}/season/{p['season_id']}"
            "/statistics/overall",
            headers=SS_HEADERS,
            tls=True,
        )
        rec[f"{label}_status"] = rs.status_code
        rec[f"{label}_pair"] = p
        if rs.status_code == 200:
            rec[f"{label}_keys"] = sorted(rs.json().get("statistics", {}).keys())
    return rec


results = [
    probe_player(p["name"]) | {"competition_id": p["competition_id"]} for p in sample_players()
]
summary = {
    "players_probed": len(results),
    "found_ids": sum("sofascore_id" in r for r in results),
    "with_seasons": sum(bool(r.get("season_pairs")) for r in results),
    "earliest_season_seen": min(
        (p["season"] for r in results for p in r.get("season_pairs", [])), default=None
    ),
    "first_full_season_by_player": {
        f"{r['competition_id']}:{r['query']}": (r.get("first_full_season") or {}).get("season")
        for r in results
    },
    "records": results,
}
write_json("sofascore_probe.json", summary)
