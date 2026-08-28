import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    text = unidecode(str(name)).lower().replace("'", "")
    text = _NON_ALNUM.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def last_token(name: str) -> str:
    tokens = normalize_name(name).split(" ")
    return tokens[-1] if tokens else ""


def _unique_lookup(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    counts = frame.groupby(keys)["right_id"].transform("nunique")
    return frame[counts == 1].drop_duplicates(keys)[keys + ["right_id"]]


MIN_FUZZY = 85  # notebook 01 Step 4: every Big-5 hit in 84–92 is a spelling variant


def match_players(
    left: pd.DataFrame, right: pd.DataFrame, *, min_fuzzy: int = MIN_FUZZY
) -> pd.DataFrame:
    left = left.copy()
    right = right.copy()
    left["_norm"] = left["name"].map(normalize_name)
    right["_norm"] = right["name"].map(normalize_name)
    left["_last"] = left["name"].map(last_token)
    right["_last"] = right["name"].map(last_token)
    left["right_id"] = pd.NA
    left["method"] = "unmatched"
    # a one-word name unique in a league-season is still a namesake risk (Step 4: 27 of 75
    # verifiable mononym hits were wrong), so name_unique only sees multi-token names
    multi_token = left["_norm"].str.contains(" ")

    stages = [
        ("exact", ["_norm", "club_key", "season"], None),
        ("last_token", ["_last", "club_key", "season"], None),
        ("name_unique", ["_norm", "season"], multi_token),
    ]
    for method, keys, eligible in stages:
        lookup = _unique_lookup(right, keys).rename(columns={"right_id": "_hit"})
        pending = left["right_id"].isna()
        if eligible is not None:
            pending &= eligible
        merged = left.loc[pending, keys].merge(lookup, on=keys, how="left")
        hits = merged["_hit"].to_numpy()
        idx = left.index[pending]
        found = pd.notna(hits)
        left.loc[idx[found], "right_id"] = hits[found]
        left.loc[idx[found], "method"] = method

    pending = left.index[left["right_id"].isna()]
    for i in pending:
        pool = right[
            (right["club_key"] == left.at[i, "club_key"])
            & (right["season"] == left.at[i, "season"])
        ]
        if pool.empty:
            continue
        target = left.at[i, "_norm"]
        scores = pool["_norm"].map(lambda s, target=target: fuzz.token_set_ratio(s, target))
        best = scores.max()
        if best >= min_fuzzy and (scores == best).sum() == 1:
            left.at[i, "right_id"] = pool.loc[scores.idxmax(), "right_id"]
            left.at[i, "method"] = "fuzzy"

    return left.drop(columns=["_norm", "_last"])


def resolve_player_ids(matches: pd.DataFrame, reep_keys: pd.Series) -> pd.DataFrame:
    """One Transfermarkt id per provider id: the reep key when there is one, else the cascade
    id that played the most minutes across the provider id's matched rows.
    `matches` = match_players output plus `provider_id` and `minutes`; `reep_keys` = provider id
    → Transfermarkt id (both as str)."""
    provider_ids = matches["provider_id"].astype(str)
    cascade = matches.assign(_pid=provider_ids).dropna(subset=["right_id"])
    weight = cascade.groupby(["_pid", "right_id"])["minutes"].sum(min_count=1).fillna(0)
    modal = weight.reset_index().sort_values("minutes", ascending=False).drop_duplicates("_pid")
    modal = modal.set_index("_pid")["right_id"].astype(int).astype(str)

    out = pd.DataFrame(index=pd.Index(provider_ids.unique(), name="provider_id"))
    out["tm_player_id"] = out.index.map(reep_keys)
    out["source"] = "reep"
    from_cascade = out["tm_player_id"].isna()
    out.loc[from_cascade, "tm_player_id"] = out.index[from_cascade].map(modal)
    out.loc[from_cascade, "source"] = "cascade"
    out.loc[out["tm_player_id"].isna(), "source"] = "unmatched"
    return out.reset_index()


OVERRIDES_DIR = Path(__file__).parent / "overrides"
TEAM_MIN_SCORE = 90  # Phase 0: resolved names scored >= 97, unresolved <= 87
LINEAGE_COLUMNS = [
    "club_id",
    "club_name",
    "competition_id",
    "provider",
    "team_name",
    "provider_team_id",
    "score",
    "source",
]


def load_overrides(name: str) -> pd.DataFrame:
    return pd.read_csv(OVERRIDES_DIR / f"{name}.csv")


def _best_club(team_name: str, choices: dict[str, tuple[int, str]]) -> tuple[float, str | None]:
    hits = process.extract(
        normalize_name(team_name), list(choices), scorer=fuzz.token_set_ratio, limit=5
    )
    if not hits:
        return 0.0, None
    score = hits[0][1]
    # token_set_ratio scores a token subset as 100 ("Barcelona" vs both "FC Barcelona" and
    # "RCD Espanyol Barcelona"); the shortest candidate among the ties is the club itself.
    return score, min((h[0] for h in hits if h[1] == score), key=len)


def build_team_lineage(
    tm_clubs: pd.DataFrame, provider_teams: dict[str, pd.DataFrame], overrides: pd.DataFrame
) -> pd.DataFrame:
    """One row per (provider, competition, team name) → Transfermarkt club_id, or NaN when the
    name scores below TEAM_MIN_SCORE. Overrides (committed CSV) win over the fuzzy match."""
    rows = []
    for provider, teams in provider_teams.items():
        for competition_id, group in teams.groupby("competition_id"):
            clubs = tm_clubs[tm_clubs.competition_id == competition_id]
            choices = {
                normalize_name(name): (club_id, name)
                for club_id, name in zip(clubs.club_id, clubs.club_name, strict=True)
            }
            names_by_id = clubs.set_index("club_id").club_name
            for record in group.itertuples(index=False):
                provider_team_id = getattr(record, "provider_team_id", None)
                override = overrides[
                    (overrides.provider == provider)
                    & (overrides.competition_id == competition_id)
                    & (overrides.team_name == record.team_name)
                ]
                if not override.empty:
                    club_id = int(override.club_id.iloc[0])
                    rows.append(
                        (
                            club_id,
                            names_by_id.get(club_id),
                            competition_id,
                            provider,
                            record.team_name,
                            provider_team_id,
                            100.0,
                            "override",
                        )
                    )
                    continue
                score, norm = _best_club(record.team_name, choices)
                resolved = norm is not None and score >= TEAM_MIN_SCORE
                club_id, club_name = choices[norm] if resolved else (pd.NA, None)
                rows.append(
                    (
                        club_id,
                        club_name,
                        competition_id,
                        provider,
                        record.team_name,
                        provider_team_id,
                        float(score),
                        "auto",
                    )
                )
    return pd.DataFrame(rows, columns=LINEAGE_COLUMNS)
