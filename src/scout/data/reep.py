import subprocess
from pathlib import Path

import pandas as pd

from scout.config import RAW, REEP_URL

REEP_DIR = RAW / "reep"
FILES = ("people", "names")
KEY_COLUMNS = ["key_transfermarkt", "key_sofascore", "key_understat", "key_fotmob"]


def ensure_files(directory: Path = REEP_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = directory / f"{name}.csv"
        if not path.exists():
            subprocess.run(["curl", "-sL", "-o", str(path), f"{REEP_URL}/{name}.csv"], check=True)
    return directory


def load_people(directory: Path | None = None) -> pd.DataFrame:
    directory = directory or ensure_files()
    return pd.read_csv(directory / "people.csv", dtype=str)


def transfermarkt_keys(people: pd.DataFrame, provider: str) -> pd.Series:
    """provider id → Transfermarkt id, for people carrying both keys. reep repeats a few
    Transfermarkt keys across rows (15 in v0); the first row wins."""
    keys = people[[f"key_{provider}", "key_transfermarkt"]].dropna()
    keys = keys.drop_duplicates(f"key_{provider}")
    return keys.set_index(f"key_{provider}")["key_transfermarkt"]
