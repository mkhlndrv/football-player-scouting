import hashlib
import json
import time
from pathlib import Path

import duckdb
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache" / "spike"
SPIKE = ROOT / "data" / "spike"
OUT = ROOT / "spike" / "out"
for _p in (RAW, CACHE, SPIKE, OUT):
    _p.mkdir(parents=True, exist_ok=True)

# Transfermarkt competition id -> soccerdata/Understat league name
BIG5 = {
    "GB1": "ENG-Premier League",
    "ES1": "ESP-La Liga",
    "IT1": "ITA-Serie A",
    "L1": "GER-Bundesliga",
    "FR1": "FRA-Ligue 1",
}
SEASONS = list(range(2014, 2026))  # season start years: 2014-15 .. 2025-26
TM_DUCKDB_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb"
)
TM_DUCKDB = RAW / "transfermarkt.duckdb"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}
MIN_GAP_S = 3.0
_last_hit = 0.0


def write_json(name: str, obj) -> Path:
    path = OUT / name
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def polite_get(
    url: str, params: dict | None = None, headers: dict | None = None
) -> requests.Response:
    global _last_hit
    key = hashlib.sha1(f"{url}|{sorted((params or {}).items())}".encode()).hexdigest()
    cached = CACHE / f"{key}.json"
    if cached.exists():
        blob = json.loads(cached.read_text())
        resp = requests.Response()
        resp.status_code = blob["status"]
        resp._content = blob["body"].encode()
        resp.url = url
        return resp
    wait = MIN_GAP_S - (time.monotonic() - _last_hit)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, params=params, headers={**HEADERS, **(headers or {})}, timeout=30)
    _last_hit = time.monotonic()
    cached.write_text(json.dumps({"status": resp.status_code, "body": resp.text}))
    return resp


def tm_connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(TM_DUCKDB), read_only=True)
