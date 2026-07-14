#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is unavailable; install/start Docker, then run make docker-smoke." >&2
  exit 2
fi

root="$(mktemp -d)"
container="multimodal-retrieval-ops-smoke-$$"
cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
  rm -rf "$root"
}
trap cleanup EXIT

python scripts/create_synthetic_service_artifacts.py "$root"
docker build --tag multimodal-retrieval-ops:smoke .
docker run --detach --name "$container" --publish 18080:8000 \
  --mount "type=bind,src=$root/artifacts,dst=/artifacts,readonly" \
  --mount "type=bind,src=$root/synthetic_manifest.csv,dst=/data/manifest.csv,readonly" \
  multimodal-retrieval-ops:smoke >/dev/null

for _ in $(seq 1 30); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18080/ready', timeout=1)" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
python -c "import json, urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:18080/health', timeout=2))['status'] == 'alive'"
python -c "import json, urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:18080/ready', timeout=2))['status'] == 'ready'"
echo "Docker smoke passed with synthetic mounted artifacts."
