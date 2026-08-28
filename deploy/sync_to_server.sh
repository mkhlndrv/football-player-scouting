#!/usr/bin/env bash
# Push the repo and the resumable pull state to the server. Usage: deploy/sync_to_server.sh ubuntu@HOST [key.pem]
set -euo pipefail
HOST="$1"; KEY="${2:-}"; SSH="ssh${KEY:+ -i $KEY}"
RSYNC=(rsync -az --info=progress2 -e "$SSH")
"${RSYNC[@]}" --exclude .venv --exclude data --exclude .git --exclude '__pycache__' --exclude .pytest_cache --exclude .ruff_cache \
  ./ "$HOST:football-player-scouting/"
# resume state: injury spells so far, and the ClubElo answers already collected (tiny; the 1 GB page cache stays here)
"${RSYNC[@]}" --relative data/raw/injuries/spells.parquet "$HOST:football-player-scouting/" || true
grep -l '"Rank,Club,Country' data/cache/http/* 2>/dev/null > /tmp/clubelo_cache_files.txt || true
"${RSYNC[@]}" --files-from=/tmp/clubelo_cache_files.txt ./ "$HOST:football-player-scouting/"
echo "synced to $HOST"
