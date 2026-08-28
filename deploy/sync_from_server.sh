#!/usr/bin/env bash
# Bring the finished pulls back. Usage: deploy/sync_from_server.sh ubuntu@HOST [key.pem]
set -euo pipefail
HOST="$1"; KEY="${2:-}"; SSH="ssh${KEY:+ -i $KEY}"
rsync -az --info=progress2 -e "$SSH" "$HOST:football-player-scouting/data/raw/injuries/" data/raw/injuries/
$SSH "$HOST" "cd football-player-scouting && grep -l '\"Rank,Club,Country' data/cache/http/*" > /tmp/clubelo_server_files.txt
rsync -az --info=progress2 -e "$SSH" --files-from=/tmp/clubelo_server_files.txt "$HOST:football-player-scouting/" ./
echo "pulled from $HOST — now: make data"
