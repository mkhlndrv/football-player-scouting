import json
from pathlib import Path

import numpy as np
import pandas as pd

from scout import config
from scout.data import fotmob, reep, understat
from scout.data import transfermarkt as tm
from scout.identity import build_team_lineage, load_overrides
from scout.models import (
    contribution,
    intervals,
    keepers,
    league_factors,
    quantities,
    recency,
    replacement,
)
from scout.panel import identity, player_match, team_season

LEAGUE_TO_COMP = {league: comp for comp, league in config.BIG5.items()}
CALIBRATION_LAST_SEASON = (
    2019  # inflation fitted up to here, coverage reported after (notebook 02 Step 9)
)
ROLE_KEYS = ["competition_id", "season", "player_id", "role"]


def _pairs(frame: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    nxt = frame.assign(season=frame["season"] - 1)[keys + [value]].rename(columns={value: "next"})
    return frame.merge(nxt, on=keys)


def _shrink_role(frame: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, dict]:
    """Shrink per90 within one role, inflation fitted on seasons <= CALIBRATION_LAST_SEASON,
    coverage reported on the later seasons. Returns the frame with point/lo/hi/k and the facts."""
    pairs = _pairs(frame, "per90", keys)
    mu, tau2 = intervals.role_prior(pairs)
    fit = intervals.shrink(frame["per90"], frame["boot_sd"], mu, tau2)
    train = pairs[pairs["season"] <= CALIBRATION_LAST_SEASON]
    fit_train = intervals.shrink(train["per90"], train["boot_sd"], mu, tau2)
    inflate = intervals.inflation(fit_train["point"], fit_train["predictive_sd"], train["next"])
    band = intervals.interval(fit["point"], fit["predictive_sd"], inflate)
    out = frame.assign(point=fit["point"], k=fit["k"], lo=band["lo"], hi=band["hi"])
    test = pairs[pairs["season"] > CALIBRATION_LAST_SEASON]
    fit_test = intervals.shrink(test["per90"], test["boot_sd"], mu, tau2)
    band_test = intervals.interval(fit_test["point"], fit_test["predictive_sd"], inflate)
    inside = (test["next"] >= band_test["lo"]) & (test["next"] <= band_test["hi"])
    facts = {
        "mu": mu,
        "tau": float(np.sqrt(tau2)),
        "inflation": inflate,
        "median_k": float(fit["k"].median()),
        "y2y_r_raw": float(pairs["per90"].corr(pairs["next"])),
        "coverage_80_out_of_sample": float(inside.mean()),
        "n_out_of_sample": int(len(test)),
        "mae_raw": float((test["per90"] - test["next"]).abs().mean()),
        "mae_shrunk": float((fit_test["point"] - test["next"]).abs().mean()),
    }
    return out, facts


def _outfield(pm: pd.DataFrame, shots: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    per90 = quantities.season_role_per90(pm, shots)
    per90 = per90[per90["minutes"] >= quantities.MIN_MINUTES].copy()
    per90["per90"] = contribution.expected_output(per90)
    matches = pm.dropna(subset=["role"]).merge(
        quantities.penalty_xg(shots), on=["game_id", "player_id"], how="left"
    )
    matches["out"] = matches["xg"] - matches["pen_xg"].fillna(0.0) + matches["xa"]
    per90 = per90.merge(
        intervals.bootstrap_sd(matches, ROLE_KEYS, "out").reset_index(), on=ROLE_KEYS
    )
    per90["ga_per90"] = per90["goals"] + per90["assists"]
    parts, facts = [], {}
    for role, group in per90.groupby("role"):
        if role == "GK":
            continue
        shrunk, role_facts = _shrink_role(group, ROLE_KEYS[:-1] + ["role"])
        ga_pairs = _pairs(group, "ga_per90", ROLE_KEYS)
        role_facts["y2y_r_goals_assists"] = float(ga_pairs["ga_per90"].corr(ga_pairs["next"]))
        parts.append(shrunk)
        facts[role] = role_facts
    table = pd.concat(parts, ignore_index=True)
    levels = replacement.replacement_level(table, "point")
    table["surplus"] = replacement.surplus(table, "point", levels)
    wide = table.pivot_table(index=["player_id", "role"], columns="season", values="point")
    history = []
    for season in sorted(table["season"].unique()):
        lags = pd.DataFrame(
            {
                f"lag{i}": wide[season - i] if (season - i) in wide.columns else np.nan
                for i in (1, 2, 3)
            },
            index=wide.index,
        )
        history.append(
            recency.weighted_history(lags)
            .rename("history_point")
            .reset_index()
            .assign(season=season)
        )
    table = table.merge(pd.concat(history), on=["player_id", "role", "season"], how="left")
    return table, {"roles": facts, "replacement_levels": levels}


def _keepers(
    pm: pd.DataFrame, shots: pd.DataFrame, team_game: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    season = keepers.prevented_per90(pm, shots, team_game)
    season = season[season["minutes"] >= quantities.MIN_MINUTES].rename(
        columns={"prevented_per90": "per90"}
    )
    sides = team_game[keepers.MATCH_KEYS + ["team_id"]]
    opponents = sides.merge(sides.rename(columns={"team_id": "opp_id"}), on=keepers.MATCH_KEYS)
    opponents = opponents[opponents["team_id"] != opponents["opp_id"]]
    faced = keepers.on_target_faced(shots).rename(columns={"team_id": "opp_id"})
    matches = keepers.keeper_matches(pm).merge(opponents, on=keepers.MATCH_KEYS + ["team_id"])
    matches = matches.merge(faced, on=keepers.MATCH_KEYS + ["opp_id"], how="left").fillna(
        {"xg_on_target": 0.0, "goals": 0}
    )
    matches["out"] = matches["xg_on_target"] - matches["goals"]
    keys = ["competition_id", "season", "player_id"]
    season = season.merge(intervals.bootstrap_sd(matches, keys, "out").reset_index(), on=keys)
    shrunk, facts = _shrink_role(season, keys)
    levels = replacement.replacement_level(shrunk.assign(role="GK"), "point")
    shrunk["surplus"] = replacement.surplus(shrunk.assign(role="GK"), "point", levels)
    return shrunk, {"GK": facts, "replacement_levels": levels}


def _fotmob_movers() -> pd.DataFrame:
    comps = list(config.BIG5) + list(config.FEEDERS)
    fm = fotmob.load()
    wide = fm[fm["stat"].isin(["_expected_goals_and_expected_assists_per_90", "mins_played"])]
    wide = wide.pivot_table(
        index=["competition_id", "season", "fotmob_player_id", "player_name", "team_name"],
        columns="stat",
        values="stat_value",
        aggfunc="first",
    ).reset_index()
    wide = wide.rename(
        columns={
            "_expected_goals_and_expected_assists_per_90": "output",
            "mins_played": "fm_minutes",
        }
    ).dropna(subset=["output", "fm_minutes"])
    wide = wide[wide["fm_minutes"] >= quantities.MIN_MINUTES]
    tm_panel = tm.load_player_club_seasons(comps, list(config.SEASONS))
    clubs = tm_panel[["club_id", "club_name", "competition_id"]].drop_duplicates()
    lineage = build_team_lineage(
        clubs,
        {"fotmob": wide[["competition_id", "team_name"]].drop_duplicates()},
        load_overrides("teams"),
    )
    resolved = identity.resolve_provider(
        "fotmob", wide, identity.transfermarkt_side(tm_panel), lineage, reep.load_people()
    )
    ids = resolved.drop_duplicates("provider_id").set_index("provider_id")["tm_player_id"]
    wide["tm_player_id"] = wide["fotmob_player_id"].astype(int).astype(str).map(ids)
    wide = (
        wide.dropna(subset=["tm_player_id"])
        .sort_values(
            ["fm_minutes", "competition_id", "fotmob_player_id"], ascending=[False, True, True]
        )
        .drop_duplicates(["season", "tm_player_id"])
    )
    players = tm.load_table("players")[["player_id", "date_of_birth"]]
    players["tm_player_id"] = players["player_id"].astype(str)
    wide = wide.merge(players[["tm_player_id", "date_of_birth"]], on="tm_player_id", how="left")
    wide["age"] = wide["season"] + 1 - pd.to_datetime(wide["date_of_birth"]).dt.year
    nxt = wide.assign(season=wide["season"] - 1)[
        ["season", "tm_player_id", "competition_id", "output"]
    ]
    nxt = nxt.rename(columns={"competition_id": "league_to", "output": "output_after"})
    movers = wide.merge(nxt, on=["season", "tm_player_id"])
    return movers[movers["competition_id"] != movers["league_to"]]


def run_contribution(models_dir: Path = config.MODELS) -> dict[str, Path]:
    pm = player_match.build()
    pm["competition_id"] = pm["league"].map(LEAGUE_TO_COMP)
    shots = understat.load("shots")
    shots["competition_id"] = shots["league"].map(LEAGUE_TO_COMP)
    team_game = team_season.team_match_long(understat.load("team_match"))
    team_game["competition_id"] = team_game["league"].map(LEAGUE_TO_COMP)

    outfield, outfield_facts = _outfield(pm, shots)
    keeper_table, keeper_facts = _keepers(pm, shots, team_game)
    factors = league_factors.tier_factors(_fotmob_movers(), set(config.BIG5))

    kill_checks = {
        "contribution_stability_beats_goals_assists": {
            role: {
                "raw": f["y2y_r_raw"],
                "goals_assists": f["y2y_r_goals_assists"],
                "pass": f["y2y_r_raw"] > f["y2y_r_goals_assists"],
            }
            for role, f in outfield_facts["roles"].items()
        },
        "contribution_variant": (
            "raw expected output; team-share and plus-minus rejected on movers (notebook 02 Step 2)"
        ),
        "defensive_value_on_off": "fails: y2y r -0.03..0.08 in every role (notebook 02 Step 6)",
        "finishing_residual": "fails persistence as expected: y2y r 0.02-0.09 (notebook 02 Step 5)",
        "opponent_slope": (
            "dropped: per-player slope not persistent (y2y r 0.01-0.10, shrinkage keeps 2-22%), "
            "adjusting movers' output changes next-club prediction by 0.000; role-average slope "
            "-0.06 (ST) / -0.045 (W) xG+xA per 90 per +100 opponent Elo (notebook 02 Step 3)"
        ),
        "keeper_proxy_stability": {
            "y2y_r": keeper_facts["GK"]["y2y_r_raw"],
            "pass": keeper_facts["GK"]["y2y_r_raw"] >= 0.3,
        },
        "interval_calibration_80": {
            role: {
                "coverage": f["coverage_80_out_of_sample"],
                "n": f["n_out_of_sample"],
                "inflation": f["inflation"],
            }
            for role, f in outfield_facts["roles"].items()
        },
        "shrinkage_improves_mae": {
            role: {
                "raw": f["mae_raw"],
                "shrunk": f["mae_shrunk"],
                "pass": f["mae_shrunk"] < f["mae_raw"],
            }
            for role, f in outfield_facts["roles"].items()
        },
        "league_factors": factors.to_dict(orient="records"),
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "contribution": models_dir / "phase2_contribution.json",
        "keepers": models_dir / "phase2_keepers.json",
        "replacement": models_dir / "phase2_replacement.json",
        "league_factors": models_dir / "phase2_league_factors.json",
        "kill_checks": models_dir / "phase2_kill_checks.json",
    }
    columns = ROLE_KEYS + ["minutes", "per90", "point", "lo", "hi", "k", "surplus", "history_point"]
    _write(paths["contribution"], outfield[columns].round(4).to_dict(orient="records"))
    keeper_columns = [
        "competition_id",
        "season",
        "player_id",
        "minutes",
        "per90",
        "point",
        "lo",
        "hi",
        "k",
        "surplus",
    ]
    _write(paths["keepers"], keeper_table[keeper_columns].round(4).to_dict(orient="records"))
    _write(
        paths["replacement"],
        {
            "outfield": outfield_facts["replacement_levels"].round(4).to_dict(orient="records"),
            "keepers": keeper_facts["replacement_levels"].round(4).to_dict(orient="records"),
            "percentile": replacement.PERCENTILE,
        },
    )
    _write(paths["league_factors"], factors.round(4).to_dict(orient="records"))
    _write(
        paths["kill_checks"],
        {"roles": {**outfield_facts["roles"], **{"GK": keeper_facts["GK"]}}, "checks": kill_checks},
    )
    return paths


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(_clean(payload), indent=1, default=_json_default))


def _clean(value):
    """JSON has no NaN: a missing number is null (pandas' NaN is a plain float, which json.dumps
    would otherwise write as the non-standard token NaN)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value))
