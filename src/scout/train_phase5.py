"""Phase 5 driver: departure cases, frozen pools, tournament verdicts (notebook 05)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scout import backtest, config
from scout.data import clubelo, reep, sofascore, understat
from scout.data import transfermarkt as tm
from scout.identity import build_team_lineage, load_overrides
from scout.models import availability, quantities, similarity
from scout.models import fit as fit_model
from scout.panel import elo as elo_panel
from scout.panel import identity, player_match, stints, workrate
from scout.panel import market as market_panel
from scout.train import _write

COMPS = list(config.BIG5) + list(config.FEEDERS)
LEAGUE_TO_COMP = {league: comp for comp, league in config.BIG5.items()}
SEASONS_RANGE = (2015, 2024)
DEF_ACTIONS = ["tackles", "interceptions", "clearances"]
Z80 = 1.2816


def _identity_bridge(tm_panel, tm_clubs, us):
    lineage = build_team_lineage(
        tm_clubs,
        {
            "understat": us[["competition_id", "team"]]
            .drop_duplicates()
            .rename(columns={"team": "team_name"})
        },
        load_overrides("teams"),
    )
    resolved = identity.resolve_provider(
        "understat", us, identity.transfermarkt_side(tm_panel), lineage, reep.load_people()
    )
    return lineage, resolved.drop_duplicates("provider_id").set_index("provider_id")["tm_player_id"]


def _universe(us_ids):
    contrib = pd.DataFrame(json.loads((config.MODELS / "phase2_contribution.json").read_text()))
    keepers = pd.DataFrame(json.loads((config.MODELS / "phase2_keepers.json").read_text()))
    keepers["role"] = "GK"
    rows = pd.concat([contrib, keepers], ignore_index=True)
    rows["tm_player_id"] = rows["player_id"].astype(int).astype(str).map(us_ids)
    rows = rows.dropna(subset=["tm_player_id"])
    return rows[
        [
            "competition_id",
            "season",
            "player_id",
            "tm_player_id",
            "role",
            "minutes",
            "per90",
            "point",
            "lo",
            "hi",
            "surplus",
        ]
    ]


def _cases(con, universe, tm_panel):
    moves = market_panel.add_cost(market_panel.classify(con))
    moves["transfer_season"] = 2000 + moves["transfer_season"].str.slice(0, 2).astype(int)
    moves["month"] = pd.to_datetime(moves["transfer_date"]).dt.month
    big5_clubs = con.execute(
        "SELECT CAST(club_id AS INTEGER) AS club_id, domestic_competition_id AS competition_id "
        "FROM clubs WHERE domestic_competition_id IN ('GB1','ES1','IT1','L1','FR1')"
    ).df()
    sales = moves[
        (moves["kind"] == "paid")
        & moves["from_club_id"].isin(big5_clubs["club_id"])
        & moves["transfer_season"].between(*SEASONS_RANGE)
        & ~moves["month"].isin([1, 2, 3])
    ].copy()
    sales["prev_season"] = sales["transfer_season"] - 1
    sales["tm_player_id"] = sales["player_id"].astype(str)
    role_of = (
        universe.sort_values(
            ["minutes", "competition_id", "player_id"], ascending=[False, True, True]
        )
        .drop_duplicates(["tm_player_id", "season"])[["tm_player_id", "season", "role"]]
        .rename(columns={"season": "prev_season", "role": "dep_role"})
    )
    cases = sales.merge(role_of, on=["tm_player_id", "prev_season"]).dropna(subset=["cost"])
    panel_clubs = tm_panel[["club_id", "season"]].drop_duplicates()
    in_panel = cases.merge(
        panel_clubs.rename(columns={"club_id": "from_club_id", "season": "transfer_season"}),
        on=["from_club_id", "transfer_season"],
    )
    incoming = moves[
        moves["kind"].isin(["paid", "free", "undisclosed"])
        & moves["to_club_id"].isin(big5_clubs["club_id"])
        & moves["transfer_season"].between(*SEASONS_RANGE)
        & ~moves["month"].isin([1, 2, 3])
    ].copy()
    incoming["tm_player_id"] = incoming["player_id"].astype(str)
    return in_panel, incoming, role_of


def _arrivals(incoming, role_of):
    both = incoming.merge(
        role_of.rename(columns={"dep_role": "in_role", "prev_season": "transfer_season"}),
        on=["tm_player_id", "transfer_season"],
        how="left",
    )
    prev = role_of.rename(columns={"dep_role": "in_role_prev"}).assign(
        transfer_season=lambda d: d["prev_season"] + 1
    )
    both = both.merge(
        prev[["tm_player_id", "transfer_season", "in_role_prev"]],
        on=["tm_player_id", "transfer_season"],
        how="left",
    )
    both["role_known"] = both["in_role_prev"].fillna(both["in_role"])
    kept = both.dropna(subset=["role_known", "cost"])[
        ["to_club_id", "transfer_season", "role_known", "tm_player_id", "cost"]
    ]
    return kept.rename(columns={"to_club_id": "from_club_id", "role_known": "dep_role"}).assign(
        from_club_id=lambda d: d["from_club_id"].astype("int64")
    )


def _club_tiers(tm_panel, season_value):
    squad = (
        season_value.groupby(["competition_id", "season", "club_id"])["value_july"]
        .sum()
        .reset_index()
    )
    squad = squad[squad["competition_id"].isin(config.BIG5)]
    elo_names = elo_panel.club_elo_names(list(config.BIG5), list(config.SEASONS))
    rows = []
    for club_id, name in elo_names.items():
        history = clubelo.fetch_club(name)
        if history.empty:
            continue
        mine = squad[squad["club_id"] == club_id]
        if mine.empty:
            continue
        dates = pd.Series([f"{s}-07-01" for s in mine["season"]])
        for (_, row), value in zip(
            mine.iterrows(), elo_panel.elo_on_dates(history, dates), strict=True
        ):
            rows.append((row["competition_id"], row["season"], club_id, value))
    club_elo = pd.DataFrame(rows, columns=["competition_id", "season", "club_id", "elo_july"])
    club_elo["elo_rank"] = club_elo.groupby(["competition_id", "season"])["elo_july"].rank(
        ascending=False
    )
    return club_elo


def _quality(ss, ss_ids, season_value):
    fields = list(backtest.QUALITY_SIGNS)
    q = ss[["competition_id", "season", "sofascore_player_id", "minutesPlayed"] + fields].copy()
    q["minutes"] = pd.to_numeric(q["minutesPlayed"])
    q["tm_player_id"] = q["sofascore_player_id"].astype(int).astype(str).map(ss_ids)
    q = q.dropna(subset=["tm_player_id"])
    for column in fields:
        q[column] = pd.to_numeric(q[column], errors="coerce")
        if "Percentage" not in column:
            q[column] = q[column] / q["minutes"] * 90
    q = q[q["minutes"] >= quantities.MIN_MINUTES].sort_values(
        ["minutes", "competition_id", "sofascore_player_id"], ascending=[False, True, True]
    )
    return q.drop_duplicates(["tm_player_id", "season"])[["tm_player_id", "season"] + fields]


DUEL_FIELDS = {
    "groundDuelsWonPercentage": "ground_duels_pct",
    "aerialDuelsWonPercentage": "aerial_duels_pct",
    "dribbledPast": "dribbled_past",
    "possessionLost": "possession_lost",
}


def defensive_traits(ss: pd.DataFrame, ss_ids: pd.Series, us_ids: pd.Series) -> pd.DataFrame:
    """Sofascore defensive traits per Understat player-season, for the similarity filter."""
    actions = ["tackles", "interceptions", "clearances"]
    rows = pd.concat(
        [
            ss[["competition_id", "season", "sofascore_player_id", "minutesPlayed"]],
            ss[list(DUEL_FIELDS)].rename(columns=DUEL_FIELDS),
            workrate.sofascore_per90(ss)[actions],
        ],
        axis=1,
    )
    rows["minutes"] = pd.to_numeric(rows["minutesPlayed"])
    rows["tm_player_id"] = rows["sofascore_player_id"].astype(int).astype(str).map(ss_ids)
    rows = rows.dropna(subset=["tm_player_id"])
    for column in DUEL_FIELDS.values():
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
        if "pct" not in column:
            rows[column] = rows[column] / rows["minutes"] * 90
    rows = rows[rows["minutes"] >= quantities.MIN_MINUTES].sort_values(
        ["minutes", "competition_id", "sofascore_player_id"], ascending=[False, True, True]
    )
    rows = rows.drop_duplicates(["tm_player_id", "season"])
    reverse = pd.Series(us_ids.index.astype(int).values, index=us_ids.to_numpy())
    reverse = reverse[~reverse.index.duplicated(keep="first")]
    rows["player_id"] = rows["tm_player_id"].map(reverse)
    kept = rows.dropna(subset=["player_id"]).copy()
    kept["player_id"] = kept["player_id"].astype(int)
    return kept[["player_id", "season", *similarity.TRAITS]]


def run_backtest(models_dir: Path = config.MODELS) -> Path:
    tm_panel = tm.load_player_club_seasons(COMPS, list(config.SEASONS))
    tm_clubs = tm_panel[["club_id", "club_name", "competition_id"]].drop_duplicates()
    us = understat.load("player_season")
    us["competition_id"] = us["league"].map(LEAGUE_TO_COMP)
    lineage, us_ids = _identity_bridge(tm_panel, tm_clubs, us)
    universe = _universe(us_ids)
    st = stints.build(COMPS, list(config.SEASONS))
    st["tm_player_id"] = st["tm_player_id"].astype(str)
    season_value = st.sort_values(["minutes", "club_id"], ascending=[False, True]).drop_duplicates(
        ["tm_player_id", "season"]
    )[["tm_player_id", "season", "club_id", "competition_id", "value_july"]]
    con = tm.connect()
    cases, incoming, role_of = _cases(con, universe, tm_panel)
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
    players = tm.load_table("players")[["player_id", "date_of_birth", "name"]]
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

    ss = sofascore.load()
    ss_lineage = build_team_lineage(
        tm_clubs,
        {"sofascore": ss[["competition_id", "team_name"]].drop_duplicates()},
        load_overrides("teams"),
    )
    ss_ids = (
        identity.resolve_provider(
            "sofascore", ss, identity.transfermarkt_side(tm_panel), ss_lineage, reep.load_people()
        )
        .drop_duplicates("provider_id")
        .set_index("provider_id")["tm_player_id"]
    )
    wr = pd.concat(
        [
            ss[["competition_id", "season", "sofascore_player_id", "minutesPlayed"]],
            workrate.sofascore_per90(ss),
        ],
        axis=1,
    )
    wr["tm_player_id"] = wr["sofascore_player_id"].astype(int).astype(str).map(ss_ids)
    wr_def = (
        wr.dropna(subset=["tm_player_id"])
        .assign(minutes=lambda d: pd.to_numeric(d["minutesPlayed"]))
        .sort_values(
            ["minutes", "competition_id", "sofascore_player_id"], ascending=[False, True, True]
        )
        .drop_duplicates(["tm_player_id", "season"])[["tm_player_id", "season"] + DEF_ACTIONS]
    )
    ga = us.copy()
    ga["tm_player_id"] = ga["player_id"].astype(int).astype(str).map(us_ids)
    ga = (
        ga.dropna(subset=["tm_player_id"])
        .groupby(["tm_player_id", "season"])[["goals", "assists", "minutes"]]
        .sum()
        .reset_index()
    )
    ga["ga90"] = (ga["goals"] + ga["assists"]) / ga["minutes"] * 90
    quality = _quality(ss, ss_ids, season_value)
    traits = defensive_traits(ss, ss_ids, us_ids)
    price = season_value[["tm_player_id", "season", "value_july"]]

    pool_frames = []
    for summer, group in cases.groupby("transfer_season"):
        past = pm[pm["season"] < summer]
        # Defensive traits were tested here and made every measured outcome worse (see the
        # state log): style matching reproduces the departing player's defending, weaknesses
        # included. The traits stay in the cards and in the outcome column, not in the filter.
        profiles = similarity.profile(past, shots[shots["season"] < summer])
        profiles = profiles[profiles["season"] == summer - 1]
        eligible = fit_model.eligible_slots(
            fit_model.role_shares(past[past["season"] == summer - 1])
        )
        price_s = (
            price[price["season"] == summer]
            .drop_duplicates("tm_player_id")
            .set_index("tm_player_id")["value_july"]
        )
        uni_s = universe[universe["season"] == summer - 1]
        train_rows = hist[hist["season"] <= summer - 2].fillna({"target": 0.0})
        hist_s = hist[hist["season"] == summer - 1].copy()
        if len(train_rows) >= 500:
            hist_s["expected_minutes"] = availability.predict(availability.fit(train_rows), hist_s)
        else:
            hist_s["expected_minutes"] = hist_s["lag1"]
        exp_min = hist_s.drop_duplicates("player_id").set_index("player_id")["expected_minutes"]
        wr_s = wr_def[wr_def["season"] == summer - 1].set_index("tm_player_id")[DEF_ACTIONS]
        ga_s = (
            ga[ga["season"] == summer - 1]
            .drop_duplicates("tm_player_id")
            .set_index("tm_player_id")["ga90"]
        )
        for case in group.itertuples():
            me = uni_s[
                (uni_s["tm_player_id"] == case.tm_player_id) & (uni_s["role"] == case.dep_role)
            ]
            if me.empty:
                continue
            me = me.sort_values(["minutes", "competition_id"], ascending=[False, True])
            bar, my_us_id = me["point"].iloc[0], me["player_id"].iloc[0]
            ok_ids = set(eligible[eligible[f"can_{case.dep_role}"]]["player_id"])
            cand = uni_s[
                uni_s["player_id"].isin(ok_ids) & (uni_s["tm_player_id"] != case.tm_player_id)
            ]
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
            gated["age"] = summer - gated["tm_player_id"].map(birth_year)
            gated["sd"] = (gated["hi"] - gated["lo"]) / (2 * Z80)
            gated["p_bar"] = 1 - stats.norm.cdf((bar - gated["point"]) / gated["sd"])
            gated["expected_minutes"] = gated["player_id"].map(exp_min)
            for action in DEF_ACTIONS:
                gated[action] = gated["tm_player_id"].map(wr_s[action])
            gated["ga90"] = gated["tm_player_id"].map(ga_s)
            gated["case_id"] = f"{case.from_club_id}_{summer}_{case.tm_player_id}"
            gated["transfer_season"] = summer
            gated["dep_role"] = case.dep_role
            gated["sale_fee"] = case.cost
            gated["bar"] = bar
            pool_frames.append(gated)
    pools = pd.concat(pool_frames, ignore_index=True)
    pools["def_sum"] = pools[DEF_ACTIONS].sum(axis=1, min_count=len(DEF_ACTIONS))
    pools["def_z"] = pools.groupby(["dep_role", "transfer_season"])["def_sum"].transform(
        lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1.0)
    )
    pools = pools.merge(
        quality.assign(transfer_season=quality["season"] + 1).drop(columns=["season"]),
        on=["tm_player_id", "transfer_season"],
        how="left",
    )
    pools["qual_z"] = backtest.quality_z(pools)
    pools["prod_per_eur"] = (pools["surplus"] * pools["expected_minutes"] / 90) / (
        pools["value"] / 1e6
    )

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
    duels = traits.merge(
        universe[["player_id", "season", "tm_player_id", "role"]].drop_duplicates(
            ["player_id", "season"]
        ),
        on=["player_id", "season"],
        how="left",
    ).dropna(subset=["tm_player_id", "role"])
    duels["duel_raw"] = duels[["ground_duels_pct", "aerial_duels_pct"]].mean(axis=1)
    duels["duel_z"] = duels.groupby(["role", "season"])["duel_raw"].transform(
        lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1.0)
    )
    duel_next = duels.set_index(["tm_player_id", "season"])["duel_z"]

    val_next = price.drop_duplicates(["tm_player_id", "season"]).set_index(
        ["tm_player_id", "season"]
    )["value_july"]
    duel_before = duels.set_index(["tm_player_id", "season"])["duel_z"]
    pools["duel_z_before"] = duel_before.reindex(
        pd.MultiIndex.from_arrays([pools["tm_player_id"], pools["transfer_season"] - 1])
    ).to_numpy()
    pools["out_duel_z"] = duel_next.reindex(
        pd.MultiIndex.from_arrays([pools["tm_player_id"], pools["transfer_season"]])
    ).to_numpy()
    pools["out_value_next"] = val_next.reindex(
        pd.MultiIndex.from_arrays([pools["tm_player_id"], pools["transfer_season"] + 1])
    ).to_numpy()

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
    actual["out_value_next"] = val_next.reindex(
        pd.MultiIndex.from_arrays([actual["tm_player_id"], actual["transfer_season"] + 1])
    ).to_numpy()
    actual["out_duel_z"] = duel_next.reindex(
        pd.MultiIndex.from_arrays([actual["tm_player_id"], actual["transfer_season"]])
    ).to_numpy()
    actual_scores = pd.DataFrame(
        {
            "case_id": actual["case_id"],
            "minutes_per_meur": actual["out_minutes"].fillna(0) / (actual["cost"] / 1e6),
            "ga90_mean": actual["out_ga90"],
            "value_ratio": actual["out_value_next"] / actual["cost"],
            "duel_quality": actual["out_duel_z"],
        }
    ).set_index("case_id")

    def model_ordering(role):
        return backtest.FIT_ORDERING if role in backtest.FIT_ROLES else backtest.OUTPUT_ORDERING

    entrants = {"actual": actual_scores}
    for name, ordering_of, only in [
        ("model", model_ordering, None),
        ("output_all", lambda r: "o2", None),
        ("market", lambda r: "market", None),
        ("naive", lambda r: "naive", None),
        ("defence", lambda r: "defence", {"CB", "FB"}),
        ("quality", lambda r: "quality", backtest.FIT_ROLES),
        ("blend", lambda r: "blend", backtest.FIT_ROLES),
        ("prod_per_eur", lambda r: "prod_per_eur", None),
    ]:
        frames = []
        for role, group in pools.groupby("dep_role"):
            if only and role not in only:
                continue
            frames.append(backtest.scores(group, ordering_of(role)).reset_index())
        entrants[name] = pd.concat(frames).set_index("case_id")

    ids_2020 = case_meta[case_meta["transfer_season"] >= 2020].index
    small_cases = cases.assign(
        case_id=lambda d: (
            d["from_club_id"].astype("int64").astype(str)
            + "_"
            + d["transfer_season"].astype("int64").astype(str)
            + "_"
            + d["tm_player_id"]
        )
    )
    ids_small = case_meta.index[
        case_meta.index.isin(small_cases[small_cases["elo_rank"] > 6]["case_id"])
    ]

    verdicts = {}
    for role, group in case_meta.groupby("dep_role"):
        verdicts[role] = {}
        for opponent in ["actual", "market", "naive"]:
            verdicts[role][opponent] = {
                "out_of_tuning_2020_plus": backtest.verdict(
                    entrants["model"], entrants[opponent], group.index.intersection(ids_2020)
                ),
                "all_seasons": backtest.verdict(entrants["model"], entrants[opponent], group.index),
            }
    tournament = {}
    for role in sorted(backtest.FIT_ROLES):
        ids = case_meta[case_meta["dep_role"] == role].index
        tournament[role] = {
            "fit_vs_output": backtest.verdict(entrants["model"], entrants["output_all"], ids),
            "quality_vs_output": backtest.verdict(entrants["quality"], entrants["output_all"], ids),
            "blend_vs_output": backtest.verdict(entrants["blend"], entrants["output_all"], ids),
            "blend_vs_fit": backtest.verdict(entrants["blend"], entrants["model"], ids),
        }
        if role in {"CB", "FB"}:
            tournament[role]["fit_vs_defence"] = backtest.verdict(
                entrants["model"], entrants["defence"], ids
            )
    formula_all_roles = {
        role: {
            "vs_output": backtest.verdict(
                entrants["prod_per_eur"], entrants["output_all"], group.index
            ),
            "vs_actual": backtest.verdict(
                entrants["prod_per_eur"], entrants["actual"], group.index
            ),
        }
        for role, group in case_meta.groupby("dep_role")
    }
    smaller_club = {
        opponent: backtest.verdict(entrants["model"], entrants[opponent], ids_small)
        for opponent in ["actual", "market", "naive"]
    }
    defensive_quality = {}
    for role, group in case_meta.groupby("dep_role"):
        for label, name in [("model", "model"), ("formula", "prod_per_eur")]:
            ids = (
                entrants[name]
                .index.intersection(entrants["actual"].index)
                .intersection(group.index)
            )
            both = (
                pd.concat(
                    [
                        entrants[name].loc[ids, "duel_quality"],
                        entrants["actual"].loc[ids, "duel_quality"],
                    ],
                    axis=1,
                ).dropna()
                if len(ids) >= 5
                else pd.DataFrame()
            )
            if len(both) < 5:
                continue
            defensive_quality.setdefault(role, {})[label] = {
                "n": int(len(both)),
                "shortlist_mean_z": round(float(both.iloc[:, 0].mean()), 4),
                "actual_signing_mean_z": round(float(both.iloc[:, 1].mean()), 4),
                "share_of_cases_better": round(
                    float((both.iloc[:, 0] > both.iloc[:, 1]).mean()), 4
                ),
            }

    def weighted_defence(pool: pd.DataFrame, weight: float, n: int = backtest.SHORTLIST):
        usable = pool.dropna(subset=["prod_per_eur", "duel_z_before"])
        if len(usable) < n:
            return usable.head(0)
        key = (1 - weight) * (-usable["prod_per_eur"]).rank() + weight * (
            -usable["duel_z_before"]
        ).rank()
        return usable.assign(_k=key).nsmallest(n, "_k")

    weighted_rows = []
    for case_id, group in pools[pools["dep_role"].isin(sorted(backtest.FIT_ROLES))].groupby(
        "case_id"
    ):
        top = weighted_defence(group, 0.25)
        if len(top) == backtest.SHORTLIST:
            weighted_rows.append({"case_id": case_id, **backtest.score_case(top)})
    entrants["defence_weighted"] = pd.DataFrame(weighted_rows).set_index("case_id")

    # The owner's question: what if defence carries an explicit weight in the ranking?
    # No exchange rate between goals and duels exists, so the weight is not fitted. Each setting
    # is scored on both outcomes and the trade-off is published instead.
    tradeoff = {}
    defensive_pools = pools[pools["dep_role"].isin(sorted(backtest.FIT_ROLES))]
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        rows = []
        for case_id, group in defensive_pools.groupby("case_id"):
            usable = group.dropna(subset=["prod_per_eur", "duel_z_before"])
            if len(usable) < backtest.SHORTLIST:
                continue
            key = (1 - weight) * (-usable["prod_per_eur"]).rank() + weight * (
                -usable["duel_z_before"]
            ).rank()
            top = usable.assign(_k=key).nsmallest(backtest.SHORTLIST, "_k")
            rows.append(
                {
                    "case_id": case_id,
                    "role": group["dep_role"].iloc[0],
                    "production": (top["surplus"] * top["expected_minutes"] / 90).mean(),
                    "duel_quality": top["out_duel_z"].mean(),
                    "minutes_per_meur": top["out_minutes"].fillna(0).sum()
                    / (top["value"].sum() / 1e6),
                    "value_ratio": top["out_value_next"].sum() / top["value"].sum(),
                }
            )
        frame = pd.DataFrame(rows)
        tradeoff[str(weight)] = {
            role: {
                "n": int(len(group)),
                "duel_quality_delivered": round(float(group["duel_quality"].mean()), 4),
                "production_expected": round(float(group["production"].mean()), 3),
                "minutes_per_meur": round(float(group["minutes_per_meur"].mean()), 1),
                "value_ratio": round(float(group["value_ratio"].mean()), 3),
            }
            for role, group in frame.groupby("role")
        }

    weighted_verdicts = {
        role: {
            "vs_actual": backtest.verdict(
                entrants["defence_weighted"], entrants["actual"], group.index
            ),
            "vs_formula": backtest.verdict(
                entrants["defence_weighted"], entrants["prod_per_eur"], group.index
            ),
        }
        for role, group in case_meta.groupby("dep_role")
        if role in backtest.FIT_ROLES
    }

    common = entrants["model"].index.intersection(entrants["actual"].index)
    _, majority = backtest.case_wins(entrants["model"], entrants["actual"], common)
    per_season = (
        majority.groupby(case_meta.loc[majority.index, "transfer_season"])
        .agg(["mean", "size"])
        .round(4)
    )

    name_of = players.drop_duplicates("tm_player_id").set_index("tm_player_id")["name"]
    club_name = (
        con.execute("SELECT CAST(club_id AS INTEGER) AS club_id, name FROM clubs")
        .df()
        .drop_duplicates("club_id")
        .set_index("club_id")["name"]
    )
    actual_by_case = actual.set_index("case_id")
    shortlist_rows = {}
    for case_id, group in pools.groupby("case_id"):
        meta = case_meta.loc[case_id]
        entry = {
            "season": int(meta["transfer_season"]),
            "role": meta["dep_role"],
            "departed": name_of.get(case_id.split("_")[2], case_id.split("_")[2]),
            "seller": club_name.get(meta["from_club_id"], str(meta["from_club_id"])),
            "fee_eur": float(meta["sale_fee"]),
            "orderings": {},
        }
        for ordering in ["prod_per_eur", "o2", "blend"]:
            top = backtest.shortlist(group, ordering)
            entry["orderings"][ordering] = [
                {
                    "name": name_of.get(r.tm_player_id, r.tm_player_id),
                    "value_eur": float(r.value),
                    "similarity": round(float(r.dist), 2),
                    "p_bar": round(float(r.p_bar), 3),
                    "prod_per_eur": None
                    if pd.isna(r.prod_per_eur)
                    else round(float(r.prod_per_eur), 3),
                    "expected_minutes": round(float(r.expected_minutes), 0),
                    "outcome_minutes": None if pd.isna(r.out_minutes) else float(r.out_minutes),
                    "outcome_ga90": None if pd.isna(r.out_ga90) else round(float(r.out_ga90), 2),
                }
                for r in top.itertuples()
            ]
        if case_id in actual_by_case.index:
            a = actual_by_case.loc[case_id]
            entry["actual_signing"] = {
                "name": name_of.get(a["tm_player_id"], a["tm_player_id"]),
                "fee_eur": float(a["cost"]),
                "outcome_minutes": None if pd.isna(a["out_minutes"]) else float(a["out_minutes"]),
                "outcome_ga90": None if pd.isna(a["out_ga90"]) else round(float(a["out_ga90"]), 2),
            }
        shortlist_rows[case_id] = entry
    _write(models_dir / "phase5_shortlists.json", shortlist_rows)

    payload = {
        "population": {
            "cases_scored": int(pools["case_id"].nunique()),
            "pool_rows": int(len(pools)),
            "cases_with_actual_signing": int(len(actual_scores)),
            "seasons": list(SEASONS_RANGE),
            "note": "summer paid departures, >=600-min role season, top-flight Big-5 sellers; "
            "Big-5 candidate pools",
        },
        "decisions": {
            "gate": f"top-{backtest.GATE_K} most similar within role",
            "budget": "candidate value <= the departing player's sale fee",
            "orderings": {"CB/FB/CM": backtest.FIT_ORDERING, "W/ST/GK": backtest.OUTPUT_ORDERING},
            "probability": "normal on the Phase 2 interval",
            "shortlist": backtest.SHORTLIST,
        },
        "verdicts": verdicts,
        "tournament_defensive_roles": tournament,
        "smaller_club_split_elo_rank_gt_6": smaller_club,
        "formula_all_roles": formula_all_roles,
        "defensive_quality_vs_actual": defensive_quality,
        "defence_weight_tradeoff": tradeoff,
        "defence_weighted_verdicts": weighted_verdicts,
        "final_design_pending_future_confirmation": {
            "outfield": "profile gate -> (point - replacement p20) x expected minutes / price",
            "GK": "P(>= bar) on the prevented proxy (formula fails: 0.26)",
            "kept_alternatives": ["output-within-gate", "similarity+output blend"],
        },
        "per_season_vs_actual": {
            int(k): {"case_win": float(v["mean"]), "n": int(v["size"])}
            for k, v in per_season.iterrows()
        },
    }
    path = models_dir / "phase5_backtest.json"
    _write(path, payload)
    return path


STAGES = {"backtest": run_backtest}
