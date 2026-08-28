import json
from pathlib import Path

import pandas as pd

from scout import config
from scout.data import clubelo, fotmob, reep, sofascore, understat
from scout.data import transfermarkt as tm
from scout.identity import build_team_lineage, load_overrides
from scout.panel import identity, player_match, stints, workrate

REPORT_PATH = config.MODELS / "phase1_data_report.json"
BARS = {"identity_big5_min": 0.95, "minutes_within_5pct": 0.90}
LEAGUE_TO_COMP = {league: comp for comp, league in config.BIG5.items()}
ELO_COUNTRY = {
    "ENG": "GB1",
    "ESP": "ES1",
    "ITA": "IT1",
    "GER": "L1",
    "FRA": "FR1",
    "BEL": "BE1",
    "NED": "NL1",
    "POR": "PO1",
    "TUR": "TR1",
    "SUI": "C1",
    "AUT": "A1",
    "DEN": "DK1",
}


def _table(frame: pd.DataFrame) -> dict:
    return {
        str(k): {str(c): (None if pd.isna(v) else round(float(v), 4)) for c, v in row.items()}
        for k, row in frame.iterrows()
    }


def _on_disk() -> dict:
    kinds = {
        k: len(list((config.RAW / "understat" / k).glob("*.parquet"))) for k in understat.KINDS
    }
    out = {"understat_league_seasons": kinds}
    for name, folder in [("sofascore", "sofascore"), ("fotmob", "fotmob")]:
        out[f"{name}_league_seasons"] = len(list((config.RAW / folder).glob("*.parquet")))
    thin = {}
    for path in sorted((config.RAW / "fotmob").glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["stat", "fotmob_player_id"])
        players = frame[frame.stat == "mins_played"].fotmob_player_id.nunique()
        if frame.stat.nunique() < 20 or players < 200:
            thin[path.stem] = {
                "stats": int(frame.stat.nunique()),
                "players_in_mins_played": int(players),
            }
    out["fotmob_thin_league_seasons"] = thin  # Bundesliga: 3 stats, no minutes list, every season
    spells = config.RAW / "injuries" / "spells.parquet"
    out["injury_players"] = (
        int(pd.read_parquet(spells).tm_player_id.nunique()) if spells.exists() else 0
    )
    return out


def _provider_frames() -> dict[str, pd.DataFrame]:
    us = understat.load("player_season")
    us["competition_id"] = us.league.map(LEAGUE_TO_COMP)
    fm = fotmob.load()
    fm_wide = fm[fm.stat == "mins_played"].rename(columns={"stat_value": "fm_minutes"})
    fm_wide = fm_wide[
        ["competition_id", "season", "fotmob_player_id", "player_name", "team_name", "fm_minutes"]
    ]
    fm_wide = fm_wide.drop_duplicates(["competition_id", "season", "fotmob_player_id"])
    return {"understat": us, "sofascore": sofascore.load(), "fotmob": fm_wide}


def _identity(frames: dict, tm_panel: pd.DataFrame, lineage: pd.DataFrame) -> dict:
    tm_side = identity.transfermarkt_side(tm_panel)
    people = reep.load_people()
    out = {}
    for provider, rows in frames.items():
        resolved = identity.resolve_provider(provider, rows, tm_side, lineage, people)
        rate = identity.minutes_rate(resolved)
        by_id = resolved.drop_duplicates("provider_id")
        big5 = rate.loc[rate.index.isin(config.BIG5)]
        out[provider] = {
            "provider_ids": int(len(by_id)),
            "by_source": by_id.source.value_counts().to_dict(),
            "minutes_rate_overall": round(
                float(
                    resolved.minutes.where(resolved.tm_player_id.notna(), 0).sum()
                    / resolved.minutes.sum()
                ),
                4,
            ),
            "minutes_rate_big5_min": round(float(big5.min().min()), 4),
            "minutes_rate_by_league_season": _table(rate),
        }
    return out


def _minutes_reconciliation(tm_panel: pd.DataFrame, lineage: pd.DataFrame) -> dict:
    rows = player_match.build()
    rows["competition_id"] = rows.league.map(LEAGUE_TO_COMP)
    clubs = lineage[lineage.provider == "understat"][["competition_id", "team_name", "club_id"]]
    rows = rows.merge(clubs.rename(columns={"team_name": "team"}), on=["competition_id", "team"])
    keys = ["competition_id", "season", "club_id"]
    both = pd.concat(
        [
            rows.groupby(keys).minutes.sum().rename("understat"),
            tm_panel.groupby(keys).minutes.sum().rename("transfermarkt"),
        ],
        axis=1,
        join="inner",
    )
    ratio = both.transfermarkt / both.understat
    within = (ratio - 1).abs() <= 0.05
    return {
        "club_seasons": int(len(both)),
        "within_5pct_share": round(float(within.mean()), 4),
        "transfermarkt_short": int((ratio < 0.95).sum()),
        "transfermarkt_long": int((ratio > 1.05).sum()),
        "within_5pct_by_league": {
            k: round(float(v), 4) for k, v in within.groupby(level="competition_id").mean().items()
        },
    }


def _roles() -> dict:
    rows = player_match.build()
    starters = (
        rows[~rows.is_sub]
        .groupby(["league", "season", "player_id", "role"])
        .minutes.sum()
        .reset_index()
    )
    eligible = (
        starters[starters.minutes >= 900].groupby(["league", "season", "role"]).player_id.nunique()
    )
    return {
        "minutes_with_role_share": round(
            float(rows.minutes[rows.role.notna()].sum() / rows.minutes.sum()), 4
        ),
        "thinnest_role_players": int(eligible.min()),
        "league_seasons_with_all_six_roles": int(
            (eligible.groupby(["league", "season"]).size() == 6).sum()
        ),
    }


def _values(comps: list[str]) -> dict:
    built = stints.build(comps, list(config.SEASONS))
    by_season = built.groupby("season").agg(
        value_at_start=("value_at_start", lambda c: c.notna().mean()),
        value_july=("value_july", lambda c: c.notna().mean()),
    )
    return {
        "stints": int(len(built)),
        "lineup_only_share": round(float(built.minutes.isna().mean()), 4),
        "coverage_by_season": _table(by_season),
    }


def _sofascore_coverage(ss: pd.DataFrame) -> dict:
    fields = [v[0] for v in workrate.SHARED.values()] + ["possessionWonAttThird"]
    active = ss[pd.to_numeric(ss.minutesPlayed) >= 90]
    coverage = active.groupby(["competition_id", "season"])[fields].apply(
        lambda g: g.notna().mean()
    )
    first = {}
    for comp, group in coverage.ge(0.9).groupby(level="competition_id"):
        seasons = group.droplevel("competition_id")
        first[comp] = {
            f: (int(seasons[f][seasons[f]].index.min()) if seasons[f].any() else None)
            for f in fields
        }
    return {"first_season_with_90pct_coverage": first}


def _elo_july_coverage(tm_panel: pd.DataFrame, tm_clubs: pd.DataFrame) -> dict:
    listings = pd.concat(
        [clubelo.list_clubs_on(f"{s}-07-01").assign(season=s) for s in config.SEASONS],
        ignore_index=True,
    )
    elo = listings[listings.Country.isin(ELO_COUNTRY)].assign(
        competition_id=lambda d: d.Country.map(ELO_COUNTRY)
    )
    names = elo[["competition_id", "Club"]].drop_duplicates().rename(columns={"Club": "team_name"})
    lineage = build_team_lineage(tm_clubs, {"clubelo": names}, load_overrides("teams")).dropna(
        subset=["club_id"]
    )
    rated = elo.merge(
        lineage[["competition_id", "team_name", "club_id"]],
        left_on=["competition_id", "Club"],
        right_on=["competition_id", "team_name"],
    )
    club_seasons = tm_panel[["competition_id", "season", "club_id"]].drop_duplicates()
    covered = club_seasons.merge(
        rated[["competition_id", "season", "club_id"]].drop_duplicates().assign(has_elo=True),
        how="left",
    )
    covered["has_elo"] = covered.has_elo.fillna(False).astype(bool)
    return {
        "club_seasons_rated_on_1_july_by_league": {
            k: round(float(v), 4)
            for k, v in covered.groupby("competition_id").has_elo.mean().items()
        }
    }


def build_report() -> dict:
    comps = list(config.BIG5) + list(config.FEEDERS)
    tm_panel = tm.load_player_club_seasons(comps, list(config.SEASONS))
    tm_clubs = tm_panel[["club_id", "club_name", "competition_id"]].drop_duplicates()
    frames = _provider_frames()
    provider_teams = {
        "understat": frames["understat"][["competition_id", "team"]]
        .drop_duplicates()
        .rename(columns={"team": "team_name"}),
        "sofascore": frames["sofascore"][["competition_id", "team_name"]].drop_duplicates(),
        "fotmob": frames["fotmob"][["competition_id", "team_name"]].drop_duplicates(),
    }
    lineage = build_team_lineage(tm_clubs, provider_teams, load_overrides("teams"))
    report = {
        "bars": BARS,
        "on_disk": _on_disk(),
        "team_lineage_unresolved": lineage.club_id.isna()
        .groupby(lineage.provider)
        .sum()
        .astype(int)
        .to_dict(),
        "identity": _identity(frames, tm_panel, lineage),
        "minutes_reconciliation_per_match": _minutes_reconciliation(tm_panel, lineage),
        "roles": _roles(),
        "values": _values(comps),
        "sofascore_coverage": _sofascore_coverage(frames["sofascore"]),
        "opponent_elo": _elo_july_coverage(tm_panel, tm_clubs),
    }
    report["passes"] = {
        "identity_big5_min": all(
            v["minutes_rate_big5_min"] >= BARS["identity_big5_min"]
            for v in report["identity"].values()
        ),
        "minutes_within_5pct": report["minutes_reconciliation_per_match"]["within_5pct_share"]
        >= BARS["minutes_within_5pct"],
    }
    return report


def write_report(path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_report(), indent=2))
    return path
