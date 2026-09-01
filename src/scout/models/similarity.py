import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from scout.models import fit, quantities

# notebook 04 Step 3: standardised Euclidean distance within role over the Phase 2 Step 1
# quantities, shot-location summaries and role minute shares. Ten neighbours' next-season output
# adds r 0.512 -> 0.525 to a player's own history; neighbour sets overlap 12% year to year (five
# times chance). The signal-weighted, PCA-whitened variant was no better on either count.
LOCATION = ["shot_dist", "box_share", "left_share"]
FEATURES = [*quantities.UNDERSTAT, *LOCATION, *fit.ROLES]
# Defensive traits (Sofascore). They repeat year to year and survive a transfer, so they belong
# in the distance that decides who is comparable, even though they failed as a ranking key.
# Ball recoveries are excluded: Sofascore only publishes them from 2023-24.
TRAITS = [
    "tackles",
    "interceptions",
    "clearances",
    "ground_duels_pct",
    "aerial_duels_pct",
    "dribbled_past",
    "possession_lost",
]
MIN_SHOTS = 10
K = 10
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
BOX_X, BOX_Y = 0.843, (0.211, 0.789)


def shot_locations(shots: pd.DataFrame) -> pd.DataFrame:
    """Per player-season: mean distance to goal, share inside the box, share from the left;
    NaN with fewer than MIN_SHOTS shots (own goals excluded)."""
    rows = shots[shots["result"] != "Own Goal"].copy()
    rows["dist"] = np.sqrt(
        ((1 - rows["location_x"]) * PITCH_LENGTH) ** 2
        + ((rows["location_y"] - 0.5) * PITCH_WIDTH) ** 2
    )
    rows["in_box"] = (rows["location_x"] >= BOX_X) & rows["location_y"].between(*BOX_Y)
    rows["left"] = rows["location_y"] < 0.5
    keys = ["competition_id", "season", "player_id"]
    out = (
        rows.groupby(keys)
        .agg(
            shot_dist=("dist", "mean"),
            box_share=("in_box", "mean"),
            left_share=("left", "mean"),
            n_shots=("dist", "size"),
        )
        .reset_index()
    )
    out.loc[out["n_shots"] < MIN_SHOTS, LOCATION] = np.nan
    return out.drop(columns="n_shots")


def profile(
    player_match: pd.DataFrame, shots: pd.DataFrame, traits: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Standardised-within-role feature rows per player-season-role at >= MIN_MINUTES.
    `traits` is optional (player_id, season, TRAITS); missing values become the role average."""
    per90 = quantities.season_role_per90(player_match, shots)
    per90 = per90[per90["minutes"] >= quantities.MIN_MINUTES]
    keys = ["competition_id", "season", "player_id"]
    rows = per90.merge(shot_locations(shots), on=keys, how="left").merge(
        fit.role_shares(player_match), on=keys, how="left"
    )
    rows = rows.sort_values("minutes", ascending=False).drop_duplicates(
        ["player_id", "season", "role"]
    )
    columns = list(FEATURES)
    if traits is not None:
        rows = rows.merge(traits, on=["player_id", "season"], how="left")
        columns += [c for c in TRAITS if c in rows.columns]
    z = rows.copy()
    for column in columns:
        z[column] = rows.groupby("role")[column].transform(_zscore).astype(float).fillna(0.0)
    return z


def _zscore(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def feature_columns(profiles: pd.DataFrame) -> list[str]:
    """The distance columns actually present: the base features plus any defensive traits."""
    return [column for column in FEATURES + TRAITS if column in profiles.columns]


def neighbours(profiles: pd.DataFrame, k: int = K) -> pd.DataFrame:
    """The k nearest player-season-roles within the same role and season, with distances."""
    out = []
    columns = feature_columns(profiles)
    for (role, season), group in profiles.groupby(["role", "season"]):
        if len(group) <= k:
            continue
        model = NearestNeighbors(n_neighbors=k + 1).fit(group[columns].to_numpy(float))
        dist, idx = model.kneighbors(group[columns].to_numpy(float))
        ids = group["player_id"].to_numpy()
        for i, pid in enumerate(ids):
            out.append(
                {
                    "role": role,
                    "season": season,
                    "player_id": pid,
                    "neighbours": ids[idx[i, 1:]].tolist(),
                    "distances": dist[i, 1:].round(3).tolist(),
                }
            )
    return pd.DataFrame(out)
