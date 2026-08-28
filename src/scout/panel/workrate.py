import pandas as pd

# notebook 01 Step 5d: canonical metric -> (Sofascore season total, FotMob stat, FotMob form).
# 16 pairs regress at slope 0.99-1.00, r >= 0.997 on 16,069 shared player-seasons; FotMob lists
# are either season totals or per-90 rates (the other form sits in sub_stat_value).
SHARED = {
    "tackles": ("tackles", "total_tackle", "per90"),
    "interceptions": ("interceptions", "interception", "per90"),
    "recoveries": ("ballRecovery", "ball_recovery", "per90"),
    "clearances": ("clearances", "effective_clearance", "per90"),
    "dribbles": ("successfulDribbles", "won_contest", "per90"),
    "key_passes": ("keyPasses", "total_att_assist", "total"),
    "big_chances_created": ("bigChancesCreated", "big_chance_created", "total"),
    "accurate_passes": ("accuratePasses", "accurate_pass", "per90"),
    "accurate_long_balls": ("accurateLongBalls", "accurate_long_balls", "per90"),
    "xg": ("expectedGoals", "expected_goals", "total"),
    "xa": ("expectedAssists", "expected_assists", "total"),
    "goals": ("goals", "goals", "total"),
    "assists": ("assists", "goal_assist", "total"),
    "fouls": ("fouls", "fouls", "per90"),
    "saves": ("saves", "saves", "per90"),
    "goals_conceded": ("goalsConceded", "goals_conceded", "per90"),
}
# slope 0.83, r 0.86: not the same definition, so each provider keeps its own column
PROVIDER_SPECIFIC = {
    "sofascore": {"possession_won_att_third_sofascore": "possessionWonAttThird"},
    "fotmob": {"possession_won_att_third_fotmob": ("poss_won_att_3rd", "per90")},
}


def sofascore_per90(rows: pd.DataFrame) -> pd.DataFrame:
    """Canonical per-90 columns from Sofascore season totals; `minutesPlayed` is the base."""
    minutes = pd.to_numeric(rows["minutesPlayed"])
    out = pd.DataFrame(index=rows.index)
    for name, (column, _, _) in SHARED.items():
        out[name] = pd.to_numeric(rows[column]) / minutes * 90
    for name, column in PROVIDER_SPECIFIC["sofascore"].items():
        out[name] = pd.to_numeric(rows[column]) / minutes * 90
    return out


def fotmob_per90(rows: pd.DataFrame) -> pd.DataFrame:
    """Canonical per-90 columns from a wide FotMob frame (one column per stat, `fm_minutes`)."""
    minutes = pd.to_numeric(rows["fm_minutes"])
    out = pd.DataFrame(index=rows.index)
    items = [(name, stat, form) for name, (_, stat, form) in SHARED.items()]
    items += [(name, stat, form) for name, (stat, form) in PROVIDER_SPECIFIC["fotmob"].items()]
    for name, stat, form in items:
        value = pd.to_numeric(rows[stat]) if stat in rows else pd.Series(pd.NA, index=rows.index)
        out[name] = value if form == "per90" else value / minutes * 90
    return out
