import json
from pathlib import Path

import numpy as np
import pandas as pd

from scout import config
from scout.data import clubelo, reep, understat
from scout.data import transfermarkt as tm
from scout.identity import build_team_lineage, load_overrides
from scout.models import availability, market, resale, trajectory
from scout.panel import elo, identity, stints
from scout.train import _write

LEAGUE_TO_COMP = {league: comp for comp, league in config.BIG5.items()}
COMPS = list(config.BIG5) + list(config.FEEDERS)
CALIBRATION_LAST_SEASON = 2019


def _contribution_with_ids() -> pd.DataFrame:
    """Phase 2 contribution rows joined to Transfermarkt ids and dates of birth."""
    contrib = pd.DataFrame(json.loads((config.MODELS / "phase2_contribution.json").read_text()))
    tm_panel = tm.load_player_club_seasons(COMPS, list(config.SEASONS))
    clubs = tm_panel[["club_id", "club_name", "competition_id"]].drop_duplicates()
    us = understat.load("player_season")
    us["competition_id"] = us["league"].map(LEAGUE_TO_COMP)
    teams = us[["competition_id", "team"]].drop_duplicates().rename(columns={"team": "team_name"})
    lineage = build_team_lineage(clubs, {"understat": teams}, load_overrides("teams"))
    resolved = identity.resolve_provider(
        "understat", us, identity.transfermarkt_side(tm_panel), lineage, reep.load_people()
    )
    ids = resolved.drop_duplicates("provider_id").set_index("provider_id")["tm_player_id"]
    contrib["tm_player_id"] = contrib["player_id"].astype(int).astype(str).map(ids)
    players = tm.load_table("players")[["player_id", "date_of_birth"]]
    players["tm_player_id"] = players["player_id"].astype(str)
    contrib = contrib.merge(
        players[["tm_player_id", "date_of_birth"]], on="tm_player_id", how="left"
    )
    contrib["age"] = contrib["season"] + 1 - pd.to_datetime(contrib["date_of_birth"]).dt.year
    return contrib.dropna(subset=["tm_player_id"])


def _season_values() -> pd.DataFrame:
    st = stints.build(COMPS, list(config.SEASONS))
    st["tm_player_id"] = st["tm_player_id"].astype(str)
    one = st.sort_values(["minutes", "club_id"], ascending=[False, True]).drop_duplicates(
        ["tm_player_id", "season"]
    )
    return one[["tm_player_id", "season", "club_id", "value_july"]]


def _club_elo_next(rows: pd.DataFrame) -> pd.DataFrame:
    names = elo.club_elo_names(COMPS, list(config.SEASONS))
    out = []
    for (season, club_id), _ in rows.groupby(["season", "club_id"]):
        name = names.get(club_id)
        if name is None:
            continue
        rating = elo.elo_on_dates(clubelo.fetch_club(name), pd.Series([f"{season + 1}-07-01"]))[0]
        out.append((season, club_id, rating))
    return pd.DataFrame(out, columns=["season", "club_id", "club_elo_next"])


def market_rows(contrib: pd.DataFrame) -> pd.DataFrame:
    values = _season_values()
    rows = contrib.merge(values, on=["tm_player_id", "season"], how="left")
    nxt = values.assign(season=values["season"] - 1)[["tm_player_id", "season", "value_july"]]
    rows = rows.merge(
        nxt.rename(columns={"value_july": "value_next_july"}),
        on=["tm_player_id", "season"],
        how="left",
    )
    rows = rows.merge(
        _club_elo_next(rows.dropna(subset=["club_id"])), on=["season", "club_id"], how="left"
    )
    return rows


def run_market(models_dir: Path = config.MODELS) -> Path:
    rows = market_rows(_contribution_with_ids())
    prepared = market.prepare(rows)
    leagues = sorted(prepared["competition_id"].unique())
    held_out = market.leave_future_out(prepared, leagues)
    model = market.fit(prepared, leagues)
    prepared["expected_log_value"] = model.predict(market.features(prepared, leagues))
    prepared["price_gap"] = prepared["y"] - prepared["expected_log_value"]
    columns = [
        "tm_player_id",
        "player_id",
        "role",
        "season",
        "competition_id",
        "age",
        "point",
        "value_july",
        "value_next_july",
        "expected_log_value",
        "price_gap",
    ]
    path = models_dir / "phase3_market.json"
    _write(
        path,
        {
            "held_out": held_out.round(4).to_dict(orient="records"),
            "rows": prepared[columns].round(4).to_dict(orient="records"),
        },
    )
    return path


def run_trajectory(models_dir: Path = config.MODELS) -> Path:
    contrib = _contribution_with_ids()
    one = contrib.sort_values(
        ["minutes", "competition_id", "player_id"], ascending=[False, True, True]
    ).drop_duplicates(["player_id", "role", "season"])
    nxt = one.assign(season=one["season"] - 1)[["player_id", "role", "season", "point"]].rename(
        columns={"point": "point_next"}
    )
    pairs = one.merge(nxt, on=["player_id", "role", "season"])
    curve = trajectory.role_curve(pairs)
    latest = one[one["season"] == one["season"].max()].copy()
    projections = {}
    for h in trajectory.HORIZONS:
        projections[f"point_h{h}"] = trajectory.project(
            latest["point"], latest["age"], latest["role"], curve, h
        )
    latest = latest.assign(**projections)
    columns = [
        "tm_player_id",
        "player_id",
        "role",
        "season",
        "age",
        "point",
        "lo",
        "hi",
        *projections,
    ]
    path = models_dir / "phase3_trajectory.json"
    _write(
        path,
        {
            "curve": [{"role": r, "age_band": b, "delta": float(v)} for (r, b), v in curve.items()],
            "latest": latest[columns].round(4).to_dict(orient="records"),
        },
    )
    return path


def run_availability(models_dir: Path = config.MODELS) -> Path:
    contrib = _contribution_with_ids()
    minutes = contrib.sort_values(
        ["minutes", "competition_id", "player_id"], ascending=[False, True, True]
    ).drop_duplicates(["player_id", "season"])
    hist = availability.history(minutes[["player_id", "season", "minutes", "age", "role"]])
    st = stints.build(COMPS, list(config.SEASONS))
    st["tm_player_id"] = st["tm_player_id"].astype(str)
    tm_minutes = (
        st.groupby(["tm_player_id", "season"])["minutes"].sum().rename("target").reset_index()
    )
    tm_minutes["season"] = tm_minutes["season"] - 1
    ids = minutes.drop_duplicates("player_id").set_index("player_id")["tm_player_id"]
    hist["tm_player_id"] = hist["player_id"].map(ids)
    hist = hist.merge(tm_minutes, on=["tm_player_id", "season"], how="left")
    known = hist[hist["season"] < hist["season"].max()].fillna({"target": 0.0})
    held_out = availability.leave_future_out(known)
    model = availability.fit(known)
    latest = hist[hist["season"] == hist["season"].max()].copy()
    latest["expected_minutes"] = availability.predict(model, latest)
    columns = [
        "tm_player_id",
        "player_id",
        "role",
        "season",
        "age",
        "lag1",
        "lag2",
        "lag3",
        "expected_minutes",
    ]
    path = models_dir / "phase3_availability.json"
    _write(
        path,
        {
            "held_out": held_out.round(1).to_dict(orient="records"),
            "latest": latest[columns].round(1).to_dict(orient="records"),
        },
    )
    return path


def _resale_bands(
    prepared: pd.DataFrame, pairs: pd.DataFrame, values: pd.Series, leagues: list[str]
) -> tuple[dict, dict]:
    """Empirical 80% bands per horizon from base seasons 2016-17 -> 2019-20 (each with the market
    model and curve trained up to that base season), and their coverage on base season 2021-22."""
    residuals: dict[int, list] = {h: [] for h in resale.HORIZONS}
    for base in range(2016, 2020):
        model = market.fit(prepared[prepared["season"] <= base], leagues)
        curve = trajectory.role_curve(pairs[pairs["season"] <= base])
        rows = prepared[prepared["season"] == base]
        for h in resale.HORIZONS:
            median = resale.median_log_value(rows, model, curve, leagues, h)
            realised = np.log10(
                pd.Series(
                    [values.get((pid, base + h), np.nan) for pid in rows["tm_player_id"]],
                    index=rows.index,
                )
            )
            residuals[h].append(realised - median)
    bands = resale.empirical_bands({h: pd.concat(r) for h, r in residuals.items()})
    test_base = 2021
    model = market.fit(prepared[prepared["season"] <= test_base], leagues)
    curve = trajectory.role_curve(pairs[pairs["season"] <= test_base])
    rows = prepared[prepared["season"] == test_base]
    coverage = {}
    for h in resale.HORIZONS:
        median = resale.median_log_value(rows, model, curve, leagues, h)
        realised = np.log10(
            pd.Series(
                [values.get((pid, test_base + h), np.nan) for pid in rows["tm_player_id"]],
                index=rows.index,
            )
        )
        band = resale.interval(median, bands[h])
        ok = realised.notna()
        coverage[h] = {
            "n": int(ok.sum()),
            "coverage_80": float(
                ((realised[ok] >= band["lo"][ok]) & (realised[ok] <= band["hi"][ok])).mean()
            ),
            "mae_log10": float((realised[ok] - median[ok]).abs().mean()),
            "no_change_mae_log10": float((realised[ok] - rows["log_prior"][ok]).abs().mean()),
        }
    return bands, coverage


def run_resale(models_dir: Path = config.MODELS) -> Path:
    contrib = _contribution_with_ids()
    rows = market_rows(contrib)
    prepared = market.prepare(rows)
    leagues = sorted(prepared["competition_id"].unique())
    one = contrib.sort_values(
        ["minutes", "competition_id", "player_id"], ascending=[False, True, True]
    ).drop_duplicates(["player_id", "role", "season"])
    nxt = one.assign(season=one["season"] - 1)[["player_id", "role", "season", "point"]].rename(
        columns={"point": "point_next"}
    )
    pairs = one.merge(nxt, on=["player_id", "role", "season"])
    values = _season_values().set_index(["tm_player_id", "season"])["value_july"]
    bands, coverage = _resale_bands(prepared, pairs, values, leagues)
    model = market.fit(prepared, leagues)
    curve = trajectory.role_curve(pairs)
    latest_season = rows["season"].max()
    latest = rows[
        (rows["season"] == latest_season)
        & rows["value_july"].notna()
        & rows["club_elo_next"].notna()
    ].copy()
    latest["y"] = np.nan
    latest["log_prior"] = np.log10(latest["value_july"])
    latest["history_point"] = latest["history_point"].fillna(latest["point"])
    latest["elo_c"] = (latest["club_elo_next"] - market.ELO_CENTRE) / 100
    out = latest[
        ["tm_player_id", "player_id", "role", "season", "age", "point", "value_july"]
    ].copy()
    for h in resale.HORIZONS:
        median = resale.median_log_value(latest, model, curve, leagues, h)
        band = resale.interval(median, bands[h])
        out[f"resale_h{h}"] = 10**median
        out[f"resale_h{h}_lo"] = 10 ** band["lo"]
        out[f"resale_h{h}_hi"] = 10 ** band["hi"]
    path = models_dir / "phase3_resale.json"
    _write(
        path,
        {
            "bands_log10": {str(h): list(b) for h, b in bands.items()},
            "calibration_base_2021": {str(h): c for h, c in coverage.items()},
            "latest": out.round(4).to_dict(orient="records"),
        },
    )
    return path


def run_kill_checks(models_dir: Path = config.MODELS) -> Path:
    """Assemble the Phase 3 kill-check table from the stage artifacts (run after them)."""

    def load(name: str) -> dict:
        return json.loads((models_dir / name).read_text())

    market_art, availability_art, resale_art = (
        load("phase3_market.json"),
        load("phase3_availability.json"),
        load("phase3_resale.json"),
    )
    held = pd.DataFrame(market_art["held_out"])
    avail = pd.DataFrame(availability_art["held_out"])
    checks = {
        "market_held_out": {
            "rmse_log10": float(held["rmse"].mean()),
            "mae_log10": float(held["mae"].mean()),
            "mdape_eur": float(held["mdape_eur"].mean()),
            "seasons": held["season"].tolist(),
            "ols_rmse_log10_notebook": 0.194,
            "market_prior_only_rmse_log10_notebook": 0.333,
            "pass": bool(held["rmse"].mean() < 0.333),
        },
        "market_monotonicity": (
            "passes: value rises with contribution, history, minutes, club Elo and the prior; "
            "falls with age from 24 (notebook 03 Step 2)"
        ),
        "price_gaps": (
            "no persistent component: same player next season r 0.05, club mean gap r 0.12 "
            "(notebook 03 Step 3)"
        ),
        "trajectory": (
            "role curve ties no-change on held-out MAE at horizons 1-3 (0.057/0.063/0.066); "
            "player-level effect worse; kept for the age direction (notebook 03 Step 4)"
        ),
        "availability_beats_last_season": {
            "mae_model": float(np.average(avail["mae_model"], weights=avail["n"])),
            "mae_baseline": float(np.average(avail["mae_baseline"], weights=avail["n"])),
            "pass": bool(
                np.average(avail["mae_model"], weights=avail["n"])
                < np.average(avail["mae_baseline"], weights=avail["n"])
            ),
        },
        "availability_injuries": (
            "no improvement on the complete file (746.2 vs 746.0 minutes MAE, baseline 803.0; "
            "notebook 03 Step 5, re-checked 2026-08-31 on 23,837 players)"
        ),
        "resale_calibration_base_2021": resale_art["calibration_base_2021"],
    }
    path = models_dir / "phase3_kill_checks.json"
    _write(path, checks)
    return path


STAGES = {
    "market": run_market,
    "trajectory": run_trajectory,
    "availability": run_availability,
    "resale": run_resale,
    "kill_checks": run_kill_checks,
}
