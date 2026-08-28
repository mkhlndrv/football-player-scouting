import numpy as np
import pandas as pd

# notebook 02 Step 9: shrink each season toward the role mean with k = tau^2 / (tau^2 + noise),
# tau^2 from the year-to-year covariance and noise from a match-level bootstrap; the predictive
# sd carries next season's noise and is inflated per role (1.2-1.5 outfield) so that 80% of
# held-out next seasons fall inside. 95% intervals cover 89-92%: tails are heavier than normal.
Z80, Z95 = 1.2816, 1.96
MIN_SD = 1e-4


def bootstrap_sd(
    per_match: pd.DataFrame, keys: list[str], value: str, n_boot: int = 200, seed: int = 0
) -> pd.Series:
    """Sampling sd of a per-90 rate from resampling a player's matches within the season."""
    rng = np.random.default_rng(seed)

    def one(group: pd.DataFrame) -> float:
        totals, minutes = group[value].to_numpy(float), group["minutes"].to_numpy(float)
        idx = rng.integers(0, len(totals), size=(n_boot, len(totals)))
        return float(np.std(totals[idx].sum(axis=1) / minutes[idx].sum(axis=1) * 90))

    return per_match.groupby(keys).apply(one).rename("boot_sd")


def role_prior(pairs: pd.DataFrame, value: str = "per90", nxt: str = "next") -> tuple[float, float]:
    """(role mean, tau^2): the between-player variance of true talent is the covariance of the
    same player's consecutive seasons."""
    return float(pairs[value].mean()), max(float(np.cov(pairs[value], pairs[nxt])[0, 1]), 1e-6)


def shrink(per90: pd.Series, boot_sd: pd.Series, mu: float, tau2: float) -> pd.DataFrame:
    noise = boot_sd.astype(float) ** 2
    k = tau2 / (tau2 + noise)
    point = mu + k * (per90 - mu)
    predictive_sd = np.sqrt(tau2 * noise / (tau2 + noise) + noise).clip(lower=MIN_SD)
    return pd.DataFrame({"point": point, "predictive_sd": predictive_sd, "k": k})


def inflation(point: pd.Series, predictive_sd: pd.Series, realised: pd.Series) -> float:
    """The factor on the predictive sd under which 80% of realised next seasons fall inside."""
    z = ((realised - point) / predictive_sd).astype(float).replace([np.inf, -np.inf], np.nan)
    return float(np.percentile(np.abs(z.dropna()), 80) / Z80)


def interval(
    point: pd.Series, predictive_sd: pd.Series, inflate: float, z: float = Z80
) -> pd.DataFrame:
    half = z * predictive_sd * inflate
    return pd.DataFrame({"lo": point - half, "hi": point + half})
