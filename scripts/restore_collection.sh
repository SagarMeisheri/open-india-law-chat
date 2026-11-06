#!/usr/bin/env bash
# Restore one Open India Law collection into a local Qdrant from the Vaquill mirror.
#
# Usage:
#   scripts/restore_collection.sh [collection] [shards]
#   scripts/restore_collection.sh acts_india 2          # default
#   scripts/restore_collection.sh legal_corpus_v1 4     # ~273 GB, do NOT try on a laptop
#
# Env overrides:
#   QDRANT_URL     (default http://localhost:6333)
#   MIN_FREE_GB    (default 35)  refuse to start if the host disk has less free space
#   SHARD_LIST     (default all) e.g. SHARD_LIST="1" to retry only shard 1
#
# Qdrant itself downloads each snapshot from the mirror and verifies the SHA256.
set -euo pipefail

COLLECTION="${1:-acts_india}"
SHARDS="${2:-2}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
MIN_FREE_GB="${MIN_FREE_GB:-35}"
BASE="https://oss-data-in.vaquill.ai/qdrant/${COLLECTION}"
SHARD_LIST="${SHARD_LIST:-$(seq -s ' ' 0 $((SHARDS - 1)))}"

for bin in curl jq awk df; do
  command -v "$bin" >/dev/null || { echo "missing dependency: $bin" >&2; exit 1; }
done

echo "==> Qdrant at ${QDRANT_URL}"
for _ in $(seq 1 30); do
  curl -fsS "${QDRANT_URL}/readyz" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "${QDRANT_URL}/readyz" >/dev/null 2>&1 \
  || { echo "Qdrant is not ready. Run: docker compose up -d" >&2; exit 1; }

# Disk preflight: snapshot download + restored data both land in the Docker volume.
FREE_GB=$(df -Pk . | awk 'NR==2 {printf "%d", $4/1024/1024}')
echo "==> Free disk: ${FREE_GB} GB (need >= ${MIN_FREE_GB} GB)"
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
  echo "Not enough free disk space. Free some space, or override with MIN_FREE_GB=<n> (at your own risk)." >&2
  exit 1
fi

# 1. Create the collection (idempotent). Config copied from the dataset README.
if curl -fsS "${QDRANT_URL}/collections/${COLLECTION}" >/dev/null 2>&1; then
  echo "==> Collection '${COLLECTION}' already exists, skipping create"
else
  echo "==> Creating collection '${COLLECTION}' with ${SHARDS} shards"
  curl -fsS -X PUT "${QDRANT_URL}/collections/${COLLECTION}" \
    -H 'Content-Type: application/json' -d "{
      \"shard_number\": ${SHARDS},
      \"vectors\": {\"dense\": {\"size\": 1024, \"distance\": \"Cosine\", \"on_disk\": true,
        \"quantization_config\": {\"scalar\": {\"type\": \"int8\", \"quantile\": 0.99}}}},
      \"sparse_vectors\": {\"sparse\": {}}
    }" | jq -c .
fi

# 2. Recover each shard straight from the mirror.
for N in $SHARD_LIST; do
  SNAP=$(curl -fsS "${BASE}/shard-${N}/index.json" | jq -r .snapshot)
  SUM=$(curl -fsS "${BASE}/shard-${N}/${SNAP}.checksum" | awk '{print $1}')
  echo "==> Shard ${N}: ${SNAP} (sha256 ${SUM:0:12}...)"
  echo "    Qdrant is downloading and recovering; this can take a long time..."
  curl -fsS --max-time 0 -X PUT \
    "${QDRANT_URL}/collections/${COLLECTION}/shards/${N}/snapshots/recover" \
    -H 'Content-Type: application/json' \
    -d "{\"location\": \"${BASE}/shard-${N}/${SNAP}\", \"checksum\": \"${SUM}\", \"priority\": \"snapshot\"}" \
    | tee /dev/stderr | jq -e '.status == "ok" or .result == true' >/dev/null \
    || { echo "Shard ${N} recovery failed" >&2; exit 1; }
  echo "    shard ${N} done"
done

echo "==> Result"
curl -fsS "${QDRANT_URL}/collections/${COLLECTION}" \
  | jq '{status: .result.status, points_count: .result.points_count, indexed_vectors_count: .result.indexed_vectors_count}'
echo "Expected points_count for acts_india: 1098577"
