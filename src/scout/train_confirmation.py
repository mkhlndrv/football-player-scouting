"""Confirmation test: the phase 5 final design replayed once on summer 2025.

Pre-registered before any outcome was read: cases, pools, budget, gate, orderings and grading
are the frozen phase 5 protocol, applied to the one sale summer the tournament never touched.
The only new inputs are 2025-26 outcomes and 1 July 2026 market values. Runs once; the result
is reported whichever way it lands.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scout import backtest, config
from scout.data import transfermarkt as tm
from scout.data import understat
from scout.models import availability, quantities, similarity
from scout.models import fit as fit_model
from scout.panel import player_match, stints, values
from scout.train import _write
from scout.train_phase5 import (
    COMPS,
    LEAGUE_TO_COMP,
    Z80,
    _arrivals,
    _cases,
    _club_tiers,
    _identity_bridge,
    _universe,
)

SUMMER = 2025  # stats season 2024-25, outcomes 2025-26, resale value at 1 July 2026


def _july_value(ids: pd.Series, valuations: pd.DataFrame, july_of: int) -> np.ndarray:
    """Market value at 1 July `july_of` from the raw valuation table.

    Phase 5 read the same quantity from next season's stint rows; those do not exist yet for
    2026-27, so the value is taken from the valuation history directly. Same definition (last
    valuation on or before the date), broader coverage, applied equally to picks and signings.
    """
    frame = pd.DataFrame(
        {"player_id": pd.to_numeric(ids).astype("int64"), "date": f"{july_of}-07-01"}
    )
    return values.value_at(frame, "date", valuations)["value"].to_numpy()


def run_confirmation(models_dir: Path = config.MODELS) -> Path:
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
    con = tm.connect()
    cases, incoming, role_of = _cases(con, universe, tm_panel, seasons_range=(SUMMER, SUMMER))
    arrivals = _arrivals(incoming, role_of)
    tiers = _club_tiers(tm_panel, season_value)
    cases = cases.merge(
        tiers.rename(columns={"club_id": "from_club_id", "season": "transfer_season"}),
        on=["from_club_id", "transfer_season"],
        how="left",
    ).dropna(subset=["elo_rank"])

    pm = player_match.build()
    pm["competition_id"] = pm["league"].map(LEAGUE_TO_COMP)
    shots = understat.load("shots")
    shots["competition_id"] = shots["league"].map(LEAGUE_TO_COMP)
    players = tm.load_table("players", con)[["player_id", "date_of_birth", "name"]]
    players["tm_player_id"] = players["player_id"].astype(str)
    birth_year = (
        players.drop_duplicates("tm_player_id")
        .set_index("tm_player_id")["date_of_birth"]
        .pipe(pd.to_datetime)
        .dt.year
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

    ga = us.copy()
    ga["tm_player_id"] = ga["player_id"].astype(int).astype(str).map(us_ids)
    ga = (
        ga.dropna(subset=["tm_player_id"])
        .groupby(["tm_player_id", "season"])[["goals", "assists", "minutes"]]
        .sum()
        .reset_index()
    )
    ga["ga90"] = (ga["goals"] + ga["assists"]) / ga["minutes"] * 90
    price = season_value[["tm_player_id", "season", "value_july"]]

    past = pm[pm["season"] < SUMMER]
    profiles = similarity.profile(past, shots[shots["season"] < SUMMER])
    profiles = profiles[profiles["season"] == SUMMER - 1]
    eligible = fit_model.eligible_slots(fit_model.role_shares(past[past["season"] == SUMMER - 1]))
    price_s = (
        price[price["season"] == SUMMER]
        .drop_duplicates("tm_player_id")
        .set_index("tm_player_id")["value_july"]
    )
    uni_s = universe[universe["season"] == SUMMER - 1]
    train_rows = hist[hist["season"] <= SUMMER - 2].fillna({"target": 0.0})
    hist_s = hist[hist["season"] == SUMMER - 1].copy()
    hist_s["expected_minutes"] = availability.predict(availability.fit(train_rows), hist_s)
    exp_min = hist_s.drop_duplicates("player_id").set_index("player_id")["expected_minutes"]
    ga_s = (
        ga[ga["season"] == SUMMER - 1]
        .drop_duplicates("tm_player_id")
        .set_index("tm_player_id")["ga90"]
    )

    pool_frames = []
    for case in cases.itertuples():
        me = uni_s[(uni_s["tm_player_id"] == case.tm_player_id) & (uni_s["role"] == case.dep_role)]
        if me.empty:
            continue
        me = me.sort_values(["minutes", "competition_id"], ascending=[False, True])
        bar, my_us_id = me["point"].iloc[0], me["player_id"].iloc[0]
        ok_ids = set(eligible[eligible[f"can_{case.dep_role}"]]["player_id"])
        cand = uni_s[uni_s["player_id"].isin(ok_ids) & (uni_s["tm_player_id"] != case.tm_player_id)]
        cand = (
            cand.sort_values(
                ["minutes", "competition_id", "player_id"], ascending=[False, True, True]
            )
            .drop_duplicates("tm_player_id")
            .copy()
        )
        cand["value"] = cand["tm_player_id"].map(price_s)
        cand = cand.dropna(subset=["value"])
        sub = (
            profiles[profiles["role"] == case.dep_role]
            .drop_duplicates("player_id")
            .set_index("player_id")
        )
        if my_us_id not in sub.index:
            continue
        distance_columns = similarity.feature_columns(sub)
        matrix = sub[distance_columns].to_numpy(float)
        mine = sub.loc[my_us_id, distance_columns].to_numpy(float)
        distances = pd.Series(np.sqrt(((matrix - mine) ** 2).sum(axis=1)), index=sub.index)
        cand["dist"] = cand["player_id"].map(distances)
        gated = cand.dropna(subset=["dist"])
        gated = gated[gated["value"] <= case.cost].nsmallest(backtest.GATE_K, "dist").copy()
        if gated.empty:
            continue
        gated["age"] = SUMMER - gated["tm_player_id"].map(birth_year)
        gated["sd"] = (gated["hi"] - gated["lo"]) / (2 * Z80)
        gated["p_bar"] = 1 - stats.norm.cdf((bar - gated["point"]) / gated["sd"])
        gated["expected_minutes"] = gated["player_id"].map(exp_min)
        gated["ga90"] = gated["tm_player_id"].map(ga_s)
        gated["case_id"] = f"{case.from_club_id}_{SUMMER}_{case.tm_player_id}"
        gated["transfer_season"] = SUMMER
        gated["dep_role"] = case.dep_role
        gated["sale_fee"] = case.cost
        pool_frames.append(gated)

    pools = pd.concat(pool_frames, ignore_index=True)
    pools["prod_per_eur"] = (pools["surplus"] * pools["expected_minutes"] / 90) / (
        pools["value"] / 1e6
    )

    valuations = tm.load_table("player_valuations", con)[
        ["player_id", "date", "market_value_in_eur"]
    ].assign(player_id=lambda d: d["player_id"].astype("int64"))
    mins_map = tm_minutes.set_index(["tm_player_id", "season"])["target"]
    pools["out_minutes"] = mins_map.reindex(
        pd.MultiIndex.from_arrays([pools["tm_player_id"], pools["transfer_season"] - 1])
    ).to_numpy()
    ga_next = ga[ga["minutes"] >= quantities.MIN_MINUTES].set_index(["tm_player_id", "season"])[
        "ga90"
    ]
    pools["out_ga90"] = ga_next.reindex(
        pd.MultiIndex.from_arrays([pools["tm_player_id"], pools["transfer_season"]])
    ).to_numpy()
    pools["out_value_next"] = _july_value(pools["tm_player_id"], valuations, SUMMER + 1)

    case_meta = pools.drop_duplicates("case_id")[
        ["case_id", "transfer_season", "dep_role", "sale_fee"]
    ].copy()
    case_meta["from_club_id"] = case_meta["case_id"].str.split("_").str[0].astype("int64")
    case_meta = case_meta.set_index("case_id")

    actual = (
        case_meta.reset_index()
        .merge(arrivals, on=["from_club_id", "transfer_season", "dep_role"])
        .sort_values(["cost", "tm_player_id"], ascending=[False, True])
        .drop_duplicates("case_id")
    )
    actual["out_minutes"] = mins_map.reindex(
        pd.MultiIndex.from_arrays([actual["tm_player_id"], actual["transfer_season"] - 1])
    ).to_numpy()
    actual["out_ga90"] = ga_next.reindex(
        pd.MultiIndex.from_arrays([actual["tm_player_id"], actual["transfer_season"]])
    ).to_numpy()
    actual["out_value_next"] = _july_value(actual["tm_player_id"], valuations, SUMMER + 1)
    actual_scores = pd.DataFrame(
        {
            "case_id": actual["case_id"],
            "minutes_per_meur": actual["out_minutes"].fillna(0) / (actual["cost"] / 1e6),
            "ga90_mean": actual["out_ga90"],
            "value_ratio": actual["out_value_next"] / actual["cost"],
        }
    ).set_index("case_id")

    def design_ordering(role):
        return "o2" if role == "GK" else "prod_per_eur"

    entrants = {"actual": actual_scores}
    for name, ordering_of in [
        ("design", design_ordering),
        ("output_all", lambda r: "o2"),
        ("blend", lambda r: "blend"),
        ("market", lambda r: "market"),
        ("naive", lambda r: "naive"),
    ]:
        frames = []
        for role, group in pools.groupby("dep_role"):
            frames.append(backtest.scores(group, ordering_of(role)).reset_index())
        entrants[name] = pd.concat(frames).set_index("case_id")

    opponents = ["actual", "market", "naive"]
    per_role = {
        role: {
            opp: backtest.verdict(entrants["design"], entrants[opp], group.index)
            for opp in opponents
        }
        for role, group in case_meta.groupby("dep_role")
    }
    outfield_ids = case_meta[case_meta["dep_role"] != "GK"].index
    pooled = {
        "outfield": {
            opp: backtest.verdict(entrants["design"], entrants[opp], outfield_ids)
            for opp in opponents
        },
        "all_roles": {
            opp: backtest.verdict(entrants["design"], entrants[opp], case_meta.index)
            for opp in opponents
        },
    }
    context = {
        name: {
            opp: backtest.verdict(entrants[name], entrants[opp], outfield_ids) for opp in opponents
        }
        for name in ["output_all", "blend"]
    }

    name_of = players.drop_duplicates("tm_player_id").set_index("tm_player_id")["name"]
    club_name = (
        con.execute("SELECT CAST(club_id AS INTEGER) AS club_id, name FROM clubs")
        .df()
        .drop_duplicates("club_id")
        .set_index("club_id")["name"]
    )
    actual_by_case = actual.set_index("case_id")
    case_rows = {}
    for case_id, group in pools.groupby("case_id"):
        meta = case_meta.loc[case_id]
        top = backtest.shortlist(group, design_ordering(meta["dep_role"]))
        entry = {
            "season": int(meta["transfer_season"]),
            "role": meta["dep_role"],
            "departed": name_of.get(case_id.split("_")[2], case_id.split("_")[2]),
            "seller": club_name.get(meta["from_club_id"], str(meta["from_club_id"])),
            "fee_eur": float(meta["sale_fee"]),
            "shortlist": [
                {
                    "name": name_of.get(r.tm_player_id, r.tm_player_id),
                    "value_eur": float(r.value),
                    "expected_minutes": round(float(r.expected_minutes), 0),
                    "outcome_minutes": None if pd.isna(r.out_minutes) else float(r.out_minutes),
                    "outcome_ga90": None if pd.isna(r.out_ga90) else round(float(r.out_ga90), 2),
                    "outcome_value_eur": None
                    if pd.isna(r.out_value_next)
                    else float(r.out_value_next),
                }
                for r in top.itertuples()
            ],
        }
        if case_id in actual_by_case.index:
            a = actual_by_case.loc[case_id]
            entry["actual_signing"] = {
                "name": name_of.get(a["tm_player_id"], a["tm_player_id"]),
                "fee_eur": float(a["cost"]),
                "outcome_minutes": None if pd.isna(a["out_minutes"]) else float(a["out_minutes"]),
                "outcome_ga90": None if pd.isna(a["out_ga90"]) else round(float(a["out_ga90"]), 2),
                "outcome_value_eur": None
                if pd.isna(a["out_value_next"])
                else float(a["out_value_next"]),
            }
        case_rows[case_id] = entry

    payload = {
        "protocol": {
            "note": "the frozen phase 5 protocol replayed once on summer 2025: same case rules, "
            "pools, budget, gate, orderings and grading; outcomes read from 2025-26",
            "design": {"outfield": "prod_per_eur within the similarity gate", "GK": "o2"},
            "value_next_source": "value_at 1 July 2026 from the raw valuation table (2026-27 "
            "stint rows do not exist); same definition as phase 5, broader coverage, applied "
            "equally to picks and signings",
        },
        "population": {
            "summer": SUMMER,
            "cases_scored": int(pools["case_id"].nunique()),
            "pool_rows": int(len(pools)),
            "cases_with_actual_signing": int(len(actual_scores)),
        },
        "design_verdicts_per_role": per_role,
        "pooled": pooled,
        "context_not_preregistered": context,
        "cases": case_rows,
    }
    path = models_dir / "phase5_confirmation.json"
    _write(path, payload)
    return path


STAGES = {"confirmation": run_confirmation}
