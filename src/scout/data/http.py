import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode, urlparse

import requests

from scout.config import CACHE

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}
BLOCKED_STATUSES = {403, 429}  # a block is not a result: never cached, so a retry re-asks
_last_hit_by_host: dict[str, float] = {}


def _cache_key(url: str, params: dict | None) -> str:
    return hashlib.sha1(f"{url}|{sorted((params or {}).items())}".encode()).hexdigest()


def _from_cache(path: Path) -> requests.Response:
    blob = json.loads(path.read_text())
    response = requests.Response()
    response.status_code = blob["status"]
    response._content = blob["body"].encode()
    response.url = blob["url"]
    return response


def _live(
    url: str, params: dict | None, headers: dict, tls: bool, timeout_s: float
) -> requests.Response:
    if not tls:
        return requests.get(url, params=params, headers=headers, timeout=timeout_s)
    import tls_requests  # browser TLS fingerprint: Sofascore 403s plain clients

    full_url = url if not params else f"{url}?{urlencode(params)}"
    raw = tls_requests.get(full_url, headers=headers, timeout=timeout_s)
    response = requests.Response()
    response.status_code, response._content, response.url = (
        raw.status_code,
        raw.text.encode(),
        full_url,
    )
    return response


def polite_get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    *,
    tls: bool = False,
    min_gap_s: float = 3.0,
    timeout_s: float = 30,
    cache_dir: Path = CACHE / "http",
) -> requests.Response:
    """GET with a disk cache (so reruns never re-hit a site) and >= min_gap_s between live
    requests to one host. The cache lives under git-ignored data/cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{_cache_key(url, params)}.json"
    if cached.exists():
        return _from_cache(cached)
    host = urlparse(url).netloc
    wait = min_gap_s - (time.monotonic() - _last_hit_by_host.get(host, -1e9))
    if wait > 0:
        time.sleep(wait)
    response = _live(url, params, {**HEADERS, **(headers or {})}, tls, timeout_s)
    _last_hit_by_host[host] = time.monotonic()
    if not 200 <= response.status_code < 500:  # 0 = TLS client transport failure; 5xx = server
        raise ConnectionError(f"{url}: status {response.status_code}: {response.text[:120]}")
    cacheable = response.status_code not in BLOCKED_STATUSES
    if cacheable and response.text.strip():  # an empty body is a soft block, not a result
        cached.write_text(
            json.dumps({"status": response.status_code, "body": response.text, "url": url})
        )
    return response
