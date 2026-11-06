"""Hybrid (dense + BM25) retrieval from Qdrant with payload pre-filters."""

import os

QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
COLLECTION = os.getenv('QDRANT_COLLECTION', 'acts_india')
EMBED_MODEL = os.getenv('VOYAGE_MODEL', 'voyage-4')
# "bm25" = hybrid via FastEmbed BM25; "off" = dense only (use if
# scripts/probe_collection.py says the stored sparse vectors are incompatible).
SPARSE_MODE = os.getenv('SPARSE_MODE', 'bm25')

# Payload keys for pre-filters, verified against the restored acts_india data
# (scripts/probe_collection.py). Note `state` is 'central' for BOTH central Acts
# and regulator instruments (RBI, SEBI, ...); `jurisdiction` ('central' |
# 'state' | 'regulatory') and `department` tell them apart.
FIELD_JURISDICTION = os.getenv('FIELD_JURISDICTION', 'state')
FIELD_ACT_STATUS = os.getenv('FIELD_ACT_STATUS', 'act_status')
FIELD_SECTION_STATUS = os.getenv('FIELD_SECTION_STATUS', 'section_status')
FIELD_YEAR = os.getenv('FIELD_YEAR', 'year')
CENTRAL_VALUE = os.getenv('CENTRAL_VALUE', 'central')

# Display names (what the planner emits). Stored values are lowercase slugs, and
# three differ from a plain slug of the name.
STATES = [
    'Assam', 'West Bengal', 'Rajasthan', 'Maharashtra', 'Chhattisgarh',
    'Uttar Pradesh', 'Kerala', 'Tamil Nadu', 'Punjab', 'Bihar', 'Odisha',
    'Karnataka', 'Telangana', 'Chandigarh', 'Uttarakhand', 'Gujarat',
    'Madhya Pradesh', 'Haryana', 'Nagaland', 'Himachal Pradesh',
    'Andhra Pradesh', 'Manipur', 'Dadra and Nagar Haveli', 'Goa', 'Jharkhand',
    'Jammu and Kashmir', 'Tripura', 'Arunachal Pradesh', 'Sikkim', 'Meghalaya',
    'Delhi', 'Mizoram', 'Puducherry', 'Ladakh', 'Andaman and Nicobar Islands',
    'Lakshadweep',
]
_SLUG_OVERRIDES = {
    'Andaman and Nicobar Islands': 'andaman-nicobar',
    'Dadra and Nagar Haveli': 'dadra-nagar-haveli',
    'Jammu and Kashmir': 'jammu-kashmir',
}
STATE_SLUGS = {n: _SLUG_OVERRIDES.get(n, n.lower().replace(' ', '-'))
               for n in STATES}
# Stored values: act_status in_force|superseded|repealed|spent;
# section_status in_force|omitted|repealed|substituted.
STATUSES = ('in_force', 'repealed', 'spent', 'superseded')

PREFETCH_K = 40
MAX_FIELD_CHARS = 1500

_qdrant = None
_voyage = None
_bm25 = None


def normalize_jurisdictions(names):
  """Maps model-supplied names to stored values. Returns (values, rejected)."""
  by_lower = {n.lower(): slug for n, slug in STATE_SLUGS.items()}
  values, rejected = [], []
  for name in names or []:
    key = name.strip().lower()
    if key == 'central':
      values.append(CENTRAL_VALUE)
    elif key in by_lower:
      values.append(by_lower[key])
    elif key in STATE_SLUGS.values():  # already a stored slug
      values.append(key)
    elif key:
      rejected.append(name)
  return list(dict.fromkeys(values)), rejected


def build_filter(filters: dict):
  """Builds a Qdrant Filter from {'jurisdictions','status','year_from','year_to'}."""
  from qdrant_client import models

  must = []
  if filters.get('jurisdictions'):
    must.append(models.FieldCondition(
        key=FIELD_JURISDICTION,
        match=models.MatchAny(any=filters['jurisdictions'])))
  status = filters.get('status')
  if status == 'in_force':  # the Act AND the section must both be in force
    must.append(models.FieldCondition(
        key=FIELD_ACT_STATUS, match=models.MatchValue(value='in_force')))
    must.append(models.FieldCondition(
        key=FIELD_SECTION_STATUS, match=models.MatchValue(value='in_force')))
  elif status == 'repealed':  # Act repealed, or section repealed/omitted
    must.append(models.Filter(should=[
        models.FieldCondition(
            key=FIELD_ACT_STATUS, match=models.MatchValue(value='repealed')),
        models.FieldCondition(
            key=FIELD_SECTION_STATUS,
            match=models.MatchAny(any=['repealed', 'omitted'])),
    ]))
  elif status:  # 'spent' | 'superseded'
    must.append(models.FieldCondition(
        key=FIELD_ACT_STATUS, match=models.MatchValue(value=status)))
  if filters.get('year_from') or filters.get('year_to'):
    must.append(models.FieldCondition(
        key=FIELD_YEAR,
        range=models.Range(
            gte=filters.get('year_from') or None,
            lte=filters.get('year_to') or None)))
  return models.Filter(must=must) if must else None


def _clip(value):
  if isinstance(value, str) and len(value) > MAX_FIELD_CHARS:
    return value[:MAX_FIELD_CHARS] + '...[truncated]'
  return value


def hybrid_search(query: str, filters: dict, top_k: int = 6) -> list[dict]:
  """Runs the filtered hybrid search. Returns [{'id','score','fields'}]."""
  global _qdrant, _voyage, _bm25
  import voyageai
  from qdrant_client import QdrantClient, models

  if _qdrant is None:
    _qdrant = QdrantClient(url=QDRANT_URL)
  if _voyage is None:
    _voyage = voyageai.Client()  # reads VOYAGE_API_KEY

  flt = build_filter(filters)
  dense = _voyage.embed(
      [query], model=EMBED_MODEL, input_type='query').embeddings[0]
  prefetch = [models.Prefetch(
      query=dense, using='dense', limit=PREFETCH_K, filter=flt)]
  if SPARSE_MODE == 'bm25':
    if _bm25 is None:
      from fastembed import SparseTextEmbedding
      _bm25 = SparseTextEmbedding('Qdrant/bm25')
    sp = next(iter(_bm25.query_embed(query)))
    prefetch.append(models.Prefetch(
        query=models.SparseVector(
            indices=sp.indices.tolist(), values=sp.values.tolist()),
        using='sparse', limit=PREFETCH_K, filter=flt))

  points = _qdrant.query_points(
      collection_name=COLLECTION,
      prefetch=prefetch,
      query=models.FusionQuery(fusion=models.Fusion.RRF),
      limit=top_k,
      with_payload=True,
  ).points
  return [
      {
          'id': str(p.id),
          'score': round(p.score, 4),
          'fields': {k: _clip(v) for k, v in (p.payload or {}).items()},
      }
      for p in points
  ]
