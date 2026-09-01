"""Phase 6 driver: the current-season "we're losing X" shortlists, precomputed for the app."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scout import backtest, config
from scout.data import reep, sofascore, understat
from scout.data import transfermarkt as tm
from scout.identity import build_team_lineage, load_overrides
from scout.models import availability, quantities, similarity
from scout.models import fit as fit_model
from scout.panel import identity as pid
from scout.panel import player_match, stints, workrate
from scout.train import _write
from scout.train_phase5 import LEAGUE_TO_COMP, Z80, _identity_bridge, _quality, _universe

COMPS = list(config.BIG5) + list(config.FEEDERS)
DEMO_SUMMER = 2025  # the last completed season is 2024-25: the demo is a summer-2025 departure
TOP_N = 10


def run_shortlists(models_dir: Path = config.MODELS) -> Path:
    tm_panel = tm.load_player_club_seasons(COMPS, list(config.SEASONS))
    tm_clubs = tm_panel[["club_id", "club_name", "competition_id"]].drop_duplicates()
    us = understat.load("player_season")
    us["competition_id"] = us["league"].map(LEAGUE_TO_COMP)
    _, us_ids = _identity_bridge(tm_panel, tm_clubs, us)
    universe = _universe(us_ids)
    st = stints.build(COMPS, list(config.SEASONS))
    st["tm_player_id"] = st["tm_player_id"].astype(str)
    season_value = st.sort_values(["minutes", "club_id"], ascending=[False, True]).drop_duplicates(
        ["tm_player_id", "season"]
    )[["tm_player_id", "season", "club_id", "competition_id", "value_july"]]
    price = (
        season_value[season_value["season"] == DEMO_SUMMER]
        .drop_duplicates("tm_player_id")
        .set_index("tm_player_id")["value_july"]
    )
    pm = player_match.build()
    pm["competition_id"] = pm["league"].map(LEAGUE_TO_COMP)
    shots = understat.load("shots")
    shots["competition_id"] = shots["league"].map(LEAGUE_TO_COMP)
    players = tm.load_table("players")[["player_id", "date_of_birth", "name"]]
    players["tm_player_id"] = players["player_id"].astype(str)
    name_of = players.drop_duplicates("tm_player_id").set_index("tm_player_id")["name"]
    birth_year = (
        players.drop_duplicates("tm_player_id")
        .set_index("tm_player_id")["date_of_birth"]
        .pipe(pd.to_datetime)
        .dt.year
    )

    past = pm[pm["season"] < DEMO_SUMMER]
    profiles = similarity.profile(past, shots[shots["season"] < DEMO_SUMMER])
    profiles = profiles[profiles["season"] == DEMO_SUMMER - 1]
    eligible = fit_model.eligible_slots(
        fit_model.role_shares(past[past["season"] == DEMO_SUMMER - 1])
    )

    minutes_rows = (
        universe.sort_values(
            ["minutes", "competition_id", "player_id"], ascending=[False, True, True]
        )
        .drop_duplicates(["player_id", "season"])
        .copy()
    )
    minutes_rows["age"] = minutes_rows["season"] + 1 - minutes_rows["tm_player_id"].map(birth_year)
    hist = availability.history(minutes_rows[["player_id", "season", "minutes", "age", "role"]])
    tm_minutes = (
        st.groupby(["tm_player_id", "season"])["minutes"].sum().rename("target").reset_index()
    )
    tm_minutes["season"] = tm_minutes["season"] - 1
    hist["tm_player_id"] = hist["player_id"].map(
        minutes_rows.drop_duplicates("player_id").set_index("player_id")["tm_player_id"]
    )
    hist = hist.merge(tm_minutes, on=["tm_player_id", "season"], how="left")
    train_rows = hist[hist["season"] <= DEMO_SUMMER - 2].fillna({"target": 0.0})
    hist_now = hist[hist["season"] == DEMO_SUMMER - 1].copy()
    hist_now["expected_minutes"] = availability.predict(availability.fit(train_rows), hist_now)
    exp_min = hist_now.drop_duplicates("player_id").set_index("player_id")["expected_minutes"]

    con = tm.connect()
    ss = sofascore.load()
    ss_lineage = build_team_lineage(
        tm_clubs,
        {"sofascore": ss[["competition_id", "team_name"]].drop_duplicates()},
        load_overrides("teams"),
    )
    ss_ids = (
        pid.resolve_provider(
            "sofascore", ss, pid.transfermarkt_side(tm_panel), ss_lineage, reep.load_people()
        )
        .drop_duplicates("provider_id")
        .set_index("provider_id")["tm_player_id"]
    )
    quality = _quality(ss, ss_ids, season_value)
    qual_now = quality[quality["season"] == DEMO_SUMMER - 1].drop_duplicates("tm_player_id")

    per90 = quantities.season_role_per90(pm, shots)
    per90_now = per90[
        (per90["season"] == DEMO_SUMMER - 1) & (per90["minutes"] >= quantities.MIN_MINUTES)
    ]
    wr = pd.concat(
        [
            ss[["competition_id", "season", "sofascore_player_id", "minutesPlayed"]],
            workrate.sofascore_per90(ss),
        ],
        axis=1,
    )
    wr["tm_player_id"] = wr["sofascore_player_id"].astype(int).astype(str).map(ss_ids)
    wr = (
        wr.dropna(subset=["tm_player_id"])
        .assign(minutes=lambda d: pd.to_numeric(d["minutesPlayed"]))
        .rename(columns={"possession_won_att_third_sofascore": "possession_won_att_third"})
    )
    wr_now = (
        wr[(wr["season"] == DEMO_SUMMER - 1) & (wr["minutes"] >= quantities.MIN_MINUTES)]
        .sort_values(
            ["minutes", "competition_id", "sofascore_player_id"], ascending=[False, True, True]
        )
        .drop_duplicates("tm_player_id")
    )
    id_link = universe[universe["season"] == DEMO_SUMMER - 1][
        ["player_id", "tm_player_id", "role", "per90"]
    ].drop_duplicates(["tm_player_id", "role"])
    card_frame = (
        per90_now.merge(id_link, on="player_id", suffixes=("", "_link"))
        .merge(
            wr_now.drop(columns=["competition_id", "season", "minutes"]),
            on="tm_player_id",
            how="left",
        )
        .merge(qual_now.drop(columns=["season"]), on="tm_player_id", how="left")
    )
    card_frame = card_frame[card_frame["role"] == card_frame["role_link"]].drop_duplicates(
        ["tm_player_id", "role"]
    )
    import json as _json

    role_profiles = _json.loads((models_dir / "phase2_role_profiles.json").read_text())["profiles"]
    card_stats = {
        role: [s["stat"] for group in cats.values() for s in group]
        for role, cats in role_profiles.items()
    }
    stat_cols = sorted(
        {c for stats in card_stats.values() for c in stats if c in card_frame.columns}
    )
    pct_frame = card_frame.copy()
    for column in stat_cols:
        pct_frame[f"pct_{column}"] = card_frame.groupby("role")[column].rank(pct=True)
    pct_frame = pct_frame.set_index(["tm_player_id", "role"])

    uni_now = universe[universe["season"] == DEMO_SUMMER - 1]
    club_name = (
        con.execute("SELECT CAST(club_id AS INTEGER) AS club_id, name FROM clubs")
        .df()
        .drop_duplicates("club_id")
        .set_index("club_id")["name"]
    )
    club_now = (
        season_value[season_value["season"] == DEMO_SUMMER - 1]
        .drop_duplicates("tm_player_id")
        .set_index("tm_player_id")["club_id"]
    )

    demo = {}
    skipped_no_value = 0
    for departing in uni_now.itertuples():
        budget = price.get(departing.tm_player_id, np.nan)
        if pd.isna(budget):
            skipped_no_value += 1
            continue
        role = departing.role
        sub = profiles[profiles["role"] == role].drop_duplicates("player_id").set_index("player_id")
        if departing.player_id not in sub.index:
            continue
        ok_ids = set(eligible[eligible[f"can_{role}"]]["player_id"])
        cand = uni_now[
            uni_now["player_id"].isin(ok_ids) & (uni_now["tm_player_id"] != departing.tm_player_id)
        ]
        cand = (
            cand.sort_values(
                ["minutes", "competition_id", "player_id"], ascending=[False, True, True]
            )
            .drop_duplicates("tm_player_id")
            .copy()
        )
        cand["value"] = cand["tm_player_id"].map(price)
        cand["age"] = DEMO_SUMMER - cand["tm_player_id"].map(birth_year)
        cand = cand.dropna(subset=["value"])
        matrix = sub[similarity.FEATURES].to_numpy(float)
        mine = sub.loc[departing.player_id, similarity.FEATURES].to_numpy(float)
        distances = pd.Series(np.sqrt(((matrix - mine) ** 2).sum(axis=1)), index=sub.index)
        cand["dist"] = cand["player_id"].map(distances)
        gated = cand.dropna(subset=["dist"])
        gated = gated[gated["value"] <= budget].nsmallest(backtest.GATE_K, "dist").copy()
        if len(gated) < TOP_N:
            continue
        gated["sd"] = (gated["hi"] - gated["lo"]) / (2 * Z80)
        gated["p_bar"] = 1 - stats.norm.cdf((departing.point - gated["point"]) / gated["sd"])
        gated["expected_minutes"] = gated["player_id"].map(exp_min)
        gated["prod_per_eur"] = (gated["surplus"] * gated["expected_minutes"] / 90) / (
            gated["value"] / 1e6
        )
        gated = gated.merge(qual_now.drop(columns=["season"]), on="tm_player_id", how="left")
        gated["dep_role"] = role
        gated["transfer_season"] = DEMO_SUMMER
        gated["qual_z"] = backtest.quality_z(gated)
        similar = cand.dropna(subset=["dist"]).nsmallest(TOP_N, "dist")
        card = []
        key = (departing.tm_player_id, role)
        if key in pct_frame.index:
            row = pct_frame.loc[key]
            for column in card_stats.get(role, []):
                if column in stat_cols and pd.notna(row.get(column)):
                    card.append(
                        {
                            "stat": column,
                            "per90": round(float(row[column]), 2),
                            "role_percentile": round(float(row[f"pct_{column}"]), 2),
                        }
                    )
        pool_rows = [
            {
                "name": name_of.get(r.tm_player_id, r.tm_player_id),
                "age": None if pd.isna(r.age) else int(r.age),
                "value_eur": float(r.value),
                "similarity": round(float(r.dist), 2),
                "p_bar": round(float(r.p_bar), 3),
                "prod_per_eur": None
                if pd.isna(r.prod_per_eur)
                else round(float(r.prod_per_eur), 3),
                "expected_minutes": None
                if pd.isna(r.expected_minutes)
                else round(float(r.expected_minutes), 0),
                "point": round(float(r.point), 3),
                "surplus": round(float(r.surplus), 3),
                # production above a freely available player. The keeper metric is negative for
                # everyone, so point x minutes would invert their ranking; surplus does not.
                "production": None
                if pd.isna(r.expected_minutes)
                else round(float(r.surplus) * float(r.expected_minutes) / 90, 1),
            }
            for r in gated.itertuples()
        ]
        entry = {
            "role": role,
            "club": club_name.get(club_now.get(departing.tm_player_id), None),
            "value_eur": float(budget),
            "point": round(float(departing.point), 3),
            "pool": pool_rows,
            "card": card,
            "similar": [
                {
                    "name": name_of.get(r.tm_player_id, r.tm_player_id),
                    "age": None if pd.isna(r.age) else int(r.age),
                    "similarity": round(float(r.dist), 2),
                    "value_eur": float(r.value),
                }
                for r in similar.itertuples()
            ],
        }
        demo[str(name_of.get(departing.tm_player_id, departing.tm_player_id))] = entry
    payload = {
        "as_of_summer": DEMO_SUMMER,
        "players": demo,
        "skipped_no_current_value": skipped_no_value,
    }
    path = models_dir / "phase6_shortlists.json"
    _write(path, payload)
    return path


STAGES = {"shortlists": run_shortlists}
