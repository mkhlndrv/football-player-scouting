import re

import pandas as pd
from rapidfuzz import fuzz
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


def match_players(left: pd.DataFrame, right: pd.DataFrame, *, min_fuzzy: int = 92) -> pd.DataFrame:
    left = left.copy()
    right = right.copy()
    left["_norm"] = left["name"].map(normalize_name)
    right["_norm"] = right["name"].map(normalize_name)
    left["_last"] = left["name"].map(last_token)
    right["_last"] = right["name"].map(last_token)
    left["right_id"] = pd.NA
    left["method"] = "unmatched"

    stages = [
        ("exact", ["_norm", "club_key", "season"]),
        ("last_token", ["_last", "club_key", "season"]),
        ("name_unique", ["_norm", "season"]),
    ]
    for method, keys in stages:
        lookup = _unique_lookup(right, keys).rename(columns={"right_id": "_hit"})
        pending = left["right_id"].isna()
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
