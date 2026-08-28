import pandas as pd

# notebook 02 Step 2: raw expected output travels with the player (movers r 0.45 / 0.56);
# team-share and the proportional plus-minus do not, so there is no team adjustment
CORE = ("npxg", "xa")


def expected_output(per90: pd.DataFrame) -> pd.Series:
    """Contribution core per 90: non-penalty xG plus xA."""
    return per90[list(CORE)].sum(axis=1).rename("expected_output")
