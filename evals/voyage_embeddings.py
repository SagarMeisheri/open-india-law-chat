"""Voyage embeddings for Ragas, using the same model family as the stored vectors.

Ragas needs a `BaseRagasEmbeddings`. The project already uses `voyage-4` for
retrieval, so the knowledge graph's similarity links are measured in the same
vector space the retriever searches.
"""

import os

import voyageai
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.run_config import RunConfig

BATCH = 64  # texts per request (Voyage allows up to 1000, but token limits bite)


class VoyageRagasEmbeddings(BaseRagasEmbeddings):

  def __init__(self, model: str | None = None, run_config: RunConfig | None = None,
               **kwargs):
    super().__init__(**kwargs)
    # BaseRagasEmbeddings expects the subclass to set this (used for retries).
    self.set_run_config(run_config or RunConfig())
    self.model = model or os.getenv('VOYAGE_MODEL', 'voyage-4')
    # max_retries: the SDK backs off on 429s, which free-tier keys hit quickly.
    self._client = voyageai.Client(max_retries=6)
    self._aclient = voyageai.AsyncClient(max_retries=6)

  def _batches(self, texts):
    for i in range(0, len(texts), BATCH):
      yield texts[i:i + BATCH]

  def embed_query(self, text: str) -> list[float]:
    return self._client.embed(
        [text], model=self.model, input_type='query').embeddings[0]

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for batch in self._batches(texts):
      out += self._client.embed(
          batch, model=self.model, input_type='document').embeddings
    return out

  async def aembed_query(self, text: str) -> list[float]:
    r = await self._aclient.embed([text], model=self.model, input_type='query')
    return r.embeddings[0]

  async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for batch in self._batches(texts):
      r = await self._aclient.embed(
          batch, model=self.model, input_type='document')
      out += r.embeddings
    return out
