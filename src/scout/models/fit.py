import numpy as np
import pandas as pd

# notebook 04 Steps 1-2. Role fit: a switch cost cannot be separated from role output levels
# and selection (W->CM "gains" +1.0 z, CM->W "loses" -1.0 z on expected output), so a slot is
# either eligible - played it enough - or not, and eligible switches are free. Style fit: on
# 2,604 matched movers no exposure to the destination's style moves post-move output (distance
# +0.009 per sd, CI spans 0; held-out gain nil), so the distance is descriptive only.
ELIGIBLE_SHARE = 0.20
ROLES = ["GK", "CB", "FB", "CM", "W", "ST"]


def role_shares(player_match: pd.DataFrame) -> pd.DataFrame:
    """Share of a player's minutes per role in a league-season (subs carry the season role)."""
    rows = player_match.dropna(subset=["role"])
    minutes = rows.groupby(["competition_id", "season", "player_id", "role"])["minutes"].sum()
    wide = minutes.unstack("role").reindex(columns=ROLES).fillna(0.0)
    return wide.div(wide.sum(axis=1), axis=0).reset_index()


def eligible_slots(shares: pd.DataFrame, threshold: float = ELIGIBLE_SHARE) -> pd.DataFrame:
    """Per player-season, the roles he can be asked to play: his main role plus any role with at
    least `threshold` of his minutes."""
    values = shares[ROLES]
    main = values.idxmax(axis=1)
    out = values.ge(threshold)
    for i, role in enumerate(main):
        out.iloc[i, ROLES.index(role)] = True
    return pd.concat(
        [shares[["competition_id", "season", "player_id"]], out.add_prefix("can_")], axis=1
    )


def style_distance(origin: pd.DataFrame, destination: pd.DataFrame) -> pd.Series:
    """Euclidean distance between two standardised style vectors, row by row. Descriptive."""
    diff = origin.to_numpy(dtype=float) - destination.to_numpy(dtype=float)
    return pd.Series(np.sqrt(np.nansum(diff**2, axis=1)), index=origin.index, name="style_distance")
