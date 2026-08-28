import pandas as pd

# notebook 02 Step 2: raw expected output travels with the player (movers r 0.45 / 0.56);
# team-share and the proportional plus-minus do not, so there is no team adjustment
CORE = ("npxg", "xa")


def expected_output(per90: pd.DataFrame) -> pd.Series:
    """Contribution core per 90: non-penalty xG plus xA."""
    return per90[list(CORE)].sum(axis=1).rename("expected_output")


# notebook 02 Step 6: a season-level on/off (team xG against with vs without the player) has
# year-to-year r of -0.03 to 0.08 in every role, so no defensive *value* is claimed from
# outcomes. Defensive actions travel with the player (movers r 0.47-0.64) and are kept as a
# per-90 activity profile - what a defender does, not what it is worth.
DEFENSIVE_ACTIONS = ("tackles", "interceptions", "clearances", "recoveries")
