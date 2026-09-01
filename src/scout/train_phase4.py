from pathlib import Path

import pandas as pd

from scout import config
from scout.data import understat
from scout.models import fit, similarity
from scout.panel import player_match
from scout.train import _write

LEAGUE_TO_COMP = {league: comp for comp, league in config.BIG5.items()}


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    pm = player_match.build()
    pm["competition_id"] = pm["league"].map(LEAGUE_TO_COMP)
    shots = understat.load("shots")
    shots["competition_id"] = shots["league"].map(LEAGUE_TO_COMP)
    return pm, shots


def run_fit(models_dir: Path = config.MODELS) -> Path:
    pm, _ = _inputs()
    shares = fit.role_shares(pm)
    slots = fit.eligible_slots(shares)
    latest = shares["season"].max()
    rows = shares.merge(slots, on=["competition_id", "season", "player_id"])
    rows = rows[rows["season"] == latest]
    path = models_dir / "phase4_fit.json"
    _write(
        path,
        {
            "eligible_share": fit.ELIGIBLE_SHARE,
            "switch_cost": (
                "not identifiable from expected output (notebook 04 Step 1); "
                "eligible switches are free"
            ),
            "style_fit": "descriptive only: no matched-transfer effect (notebook 04 Step 2)",
            "latest": rows.round(3).to_dict(orient="records"),
        },
    )
    return path


def _traits() -> pd.DataFrame:
    """Defensive traits for the similarity filter, built exactly as the backtest builds them."""
    from scout.data import reep, sofascore
    from scout.data import transfermarkt as tm
    from scout.identity import build_team_lineage, load_overrides
    from scout.panel import identity as pid
    from scout.train_phase5 import _identity_bridge, defensive_traits

    comps = list(config.BIG5) + list(config.FEEDERS)
    tm_panel = tm.load_player_club_seasons(comps, list(config.SEASONS))
    tm_clubs = tm_panel[["club_id", "club_name", "competition_id"]].drop_duplicates()
    us = understat.load("player_season")
    us["competition_id"] = us["league"].map({league: comp for comp, league in config.BIG5.items()})
    _, us_ids = _identity_bridge(tm_panel, tm_clubs, us)
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
    return defensive_traits(ss, ss_ids, us_ids)


def run_similarity(models_dir: Path = config.MODELS) -> Path:
    pm, shots = _inputs()
    profiles = similarity.profile(pm, shots, _traits())
    latest = profiles["season"].max()
    current = profiles[profiles["season"] == latest]
    neighbours = similarity.neighbours(current)
    features = [c for c in similarity.FEATURES + similarity.TRAITS if c in current.columns]
    columns = ["competition_id", "season", "player_id", "role", "minutes", *features]
    path = models_dir / "phase4_similarity.json"
    _write(
        path,
        {
            "features": features,
            "k": similarity.K,
            "profiles": current[columns].round(3).to_dict(orient="records"),
            "neighbours": neighbours.to_dict(orient="records"),
        },
    )
    return path


def run_kill_checks(models_dir: Path = config.MODELS) -> Path:
    checks = {
        "role_switch_cost": (
            "not identifiable: within-role z flips sign between attacking and non-attacking "
            "roles (W->CM +1.04, CM->W -0.99) - role level plus selection; eligible switches "
            "treated as free (notebook 04 Step 1)"
        ),
        "style_fit_matched_transfer": {
            "movers": 2604,
            "distance_effect_per_sd": 0.0088,
            "ci80": [-0.0025, 0.0202],
            "matched_cells_effect": 0.0058,
            "matched_ci80": [-0.0227, 0.0347],
            "held_out_mae_base": 0.3187,
            "held_out_mae_best": 0.3172,
            "pass": False,
            "verdict": "descriptive only (notebook 04 Step 2)",
        },
        "similarity_validity": {
            "own_history_r": 0.512,
            "own_plus_neighbours_r": 0.525,
            "top10_overlap_year_to_year": 0.122,
            "verdict": "validated, small; descriptive lens (notebook 04 Step 3)",
        },
    }
    path = models_dir / "phase4_kill_checks.json"
    _write(path, checks)
    return path


STAGES = {"fit": run_fit, "similarity": run_similarity, "kill_checks_phase4": run_kill_checks}
