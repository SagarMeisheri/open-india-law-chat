"""Show collection status and one sample payload, to learn the real field names.

Usage: uv run python scripts/inspect_collection.py [collection]
"""

import json
import os
import sys

from qdrant_client import QdrantClient

name = sys.argv[1] if len(sys.argv) > 1 else os.getenv('QDRANT_COLLECTION', 'acts_india')
client = QdrantClient(url=os.getenv('QDRANT_URL', 'http://localhost:6333'))

info = client.get_collection(name)
print(f'collection={name} status={info.status} points={info.points_count}')
print('vectors:', list((info.config.params.vectors or {}).keys()))

points, _ = client.scroll(name, limit=1, with_payload=True, with_vectors=False)
if points:
  print('sample payload:')
  print(json.dumps(points[0].payload, indent=2, ensure_ascii=False)[:3000])
