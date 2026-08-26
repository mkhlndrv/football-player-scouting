# data/

Git-ignored except this file. Everything here is fetched by code.

- `raw/`    downloaded inputs as received (Transfermarkt DuckDB, provider snapshots)
- `cache/`  HTTP caches (soccerdata, spike probes) so reruns do not re-hit sites
- `spike/`  Phase 0 intermediate parquet files

Rebuild with `make spike` (Phase 0) or `make data` (Phase 1 onward).
