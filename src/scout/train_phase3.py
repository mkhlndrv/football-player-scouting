import json
from pathlib import Path

import pandas as pd

from scout import config
from scout.data import clubelo, reep, understat
from scout.data import transfermarkt as tm
from scout.identity import build_team_lineage, load_overrides
from scout.models import availability, market, trajectory
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
    one = st.sort_values("minutes", ascending=False).drop_duplicates(["tm_player_id", "season"])
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
    one = contrib.sort_values("minutes", ascending=False).drop_duplicates(
        ["player_id", "role", "season"]
    )
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
    minutes = contrib.sort_values("minutes", ascending=False).drop_duplicates(
        ["player_id", "season"]
    )
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


STAGES = {"market": run_market, "trajectory": run_trajectory, "availability": run_availability}
