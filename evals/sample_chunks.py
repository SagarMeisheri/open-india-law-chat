"""Draws a stratified, Act-clustered sample of provisions from Qdrant.

Why not just scroll the first N points: a scroll returns the first points by id,
not a random or representative set. And 100 random provisions out of ~1.1M
almost never share an Act, so the Ragas knowledge graph would have no
neighbours and the multi-hop synthesizers would find nothing to link.

So we (1) pick random in-force points per jurisdiction stratum, (2) keep their
distinct Acts, and (3) pull several provisions from each of those Acts.
"""

import os
import random
from dataclasses import dataclass

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
COLLECTION = os.getenv('QDRANT_COLLECTION', 'acts_india')

# Share of Acts drawn from each `jurisdiction` value (see README "Data coverage").
STRATA = {'central': 0.40, 'state': 0.40, 'regulatory': 0.20}
MIN_CHARS, MAX_CHARS = 400, 3500  # skip stubs and very long provisions

# Payload keys kept as Document metadata: they become the gold labels.
META_KEYS = (
    'chunk_id', 'act_id', 'title', 'section_number', 'section_title',
    'jurisdiction', 'state', 'department', 'year', 'act_status',
    'section_status', 'source_url', 'chunk_index',
)


@dataclass
class SampleReport:
  acts: int
  chunks: int
  per_stratum: dict


def _must(*conds):
  return models.Filter(must=list(conds))


def _match(key, value):
  return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _client() -> QdrantClient:
  return QdrantClient(url=QDRANT_URL, timeout=int(os.getenv('QDRANT_TIMEOUT', '60')),
                      check_compatibility=False)


def sample_documents(
    n_acts: int = 40,
    chunks_per_act: int = 8,
    seed: int = 7,
    client: QdrantClient | None = None,
) -> tuple[list[Document], SampleReport]:
  """Returns (documents, report). Deterministic given `seed` and the data."""
  client = client or _client()
  rng = random.Random(seed)
  docs: list[Document] = []
  per_stratum: dict[str, int] = {}
  seen_acts: set[str] = set()

  for stratum, share in STRATA.items():
    want = max(1, round(n_acts * share))
    base = [_match('jurisdiction', stratum), _match('act_status', 'in_force'),
            _match('section_status', 'in_force'), _match('section_type', 'section')]
    # Random points -> distinct Acts. Oversample: many draws land in the same Act.
    pts = client.query_points(
        COLLECTION,
        query=models.SampleQuery(sample=models.Sample.RANDOM),
        query_filter=_must(*base), limit=want * 12,
        with_payload=['act_id'], with_vectors=False).points
    act_ids = []
    for p in pts:
      aid = (p.payload or {}).get('act_id')
      if aid and aid not in seen_acts and aid not in act_ids:
        act_ids.append(aid)
    rng.shuffle(act_ids)

    got = 0
    for aid in act_ids:
      if got >= want:
        break
      records, _ = client.scroll(
          COLLECTION, scroll_filter=_must(*base, _match('act_id', aid)),
          limit=200, with_payload=True, with_vectors=False)
      usable = [r for r in records
                if MIN_CHARS <= len((r.payload or {}).get('text', '')) <= MAX_CHARS]
      if len(usable) < 3:  # too small an Act to make neighbours
        continue
      chosen = rng.sample(usable, min(chunks_per_act, len(usable)))
      chosen.sort(key=lambda r: r.payload.get('chunk_index', 0))
      for r in chosen:
        meta = {k: r.payload.get(k) for k in META_KEYS}
        docs.append(Document(page_content=r.payload['text'], metadata=meta))
      seen_acts.add(aid)
      got += 1
      per_stratum[stratum] = per_stratum.get(stratum, 0) + len(chosen)

  return docs, SampleReport(acts=len(seen_acts), chunks=len(docs),
                            per_stratum=per_stratum)
