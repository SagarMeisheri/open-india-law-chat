"""Probe a restored collection: payload fields, indexes, filter values, and
whether the stored BM25 sparse vectors are compatible with FastEmbed BM25.

Usage:
  uv run python scripts/probe_collection.py [collection]
  uv run python scripts/probe_collection.py acts_india --create-indexes
"""

import json
import os
import sys

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

args = [a for a in sys.argv[1:] if not a.startswith('--')]
name = args[0] if args else os.getenv('QDRANT_COLLECTION', 'acts_india')
client = QdrantClient(url=os.getenv('QDRANT_URL', 'http://localhost:6333'))

info = client.get_collection(name)
print(f'== {name}: status={info.status} points={info.points_count}')
print('dense/sparse:', list((info.config.params.vectors or {}).keys()),
      list((info.config.params.sparse_vectors or {}).keys()))
sparse_cfg = (info.config.params.sparse_vectors or {}).get('sparse')
print('sparse modifier (idf?):', getattr(sparse_cfg, 'modifier', None))
print('payload indexes:', {k: str(v.data_type) for k, v in info.payload_schema.items()})

points, _ = client.scroll(
    name, limit=3, with_payload=True, with_vectors=['sparse'])
if not points:
  sys.exit('collection is empty')

print('\n== sample payload keys:')
print(json.dumps(
    {k: (v[:80] + '...' if isinstance(v, str) and len(v) > 80 else v)
     for k, v in points[0].payload.items()}, indent=2, ensure_ascii=False))

# --- Filter-value discovery (needs a keyword payload index) -----------------
print('\n== distinct values (facet; needs keyword index):')
for key in ('state', 'court', 'section_status', 'act_status', 'jurisdiction'):
  try:
    hits = client.facet(name, key=key, limit=15).hits
    print(f'  {key}:', [(h.value, h.count) for h in hits])
  except Exception as exc:
    print(f'  {key}: n/a ({type(exc).__name__})')

# --- Sparse compatibility test ---------------------------------------------
print('\n== sparse compatibility with FastEmbed Qdrant/bm25:')
bm25 = SparseTextEmbedding('Qdrant/bm25')
text_key = next((k for k in ('text', 'content', 'chunk_text', 'body')
                 if k in points[0].payload), None)
if not text_key:
  print('  could not find the text field; set it by hand in this script')
else:
  for p in points:
    stored = p.vector['sparse']
    mine = next(iter(bm25.embed([p.payload[text_key]])))
    s_idx, m_idx = set(stored.indices), set(mine.indices.tolist())
    overlap = len(s_idx & m_idx) / max(1, len(s_idx | m_idx))
    print(f'  point {p.id}: stored={len(s_idx)} fastembed={len(m_idx)} '
          f'index-overlap={overlap:.0%}  stored-weights~{stored.values[:3]}')
  print('  >=90% overlap  => FastEmbed-compatible: SPARSE_MODE=bm25 will work.')
  print('  low overlap    => different tokenizer: use SPARSE_MODE=off, or add a')
  print('                    full-text payload index and filter with MatchText.')

# --- Optional: create payload indexes for fast pre-filtering ----------------
if '--create-indexes' in sys.argv:
  wanted = {'state': 'keyword', 'section_status': 'keyword',
            'act_status': 'keyword', 'court': 'keyword', 'year': 'integer'}
  for key, kind in wanted.items():
    if key in points[0].payload and key not in info.payload_schema:
      client.create_payload_index(name, key, field_schema=kind)
      print(f'created {kind} index on {key}')
