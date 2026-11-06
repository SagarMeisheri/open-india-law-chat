"""Generates a candidate golden dataset with Ragas from Qdrant provisions.

  uv run --group evals python -m evals.generate_golden --dry-run
  uv run --group evals python -m evals.generate_golden --size 5 --out evals/golden/smoke.jsonl
  uv run --group evals python -m evals.generate_golden --size 150 --acts 60

Output rows are CANDIDATES (review_status="unreviewed"). Ragas writes the
reference answer with an LLM, so a human (ideally an advocate) must review them
before they are used as ground truth. Each row also carries the gold chunk ids
and section numbers of its source provisions, so retrieval can be scored
without an LLM.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

from . import sample_chunks  # noqa: E402

LLM_CONTEXT = (
    'The source texts are provisions of Indian Acts and regulator instruments '
    '(Central, State and RBI/SEBI-type regulations). Write questions the way a '
    'real user would ask them. Every question must be answerable from the given '
    'provision text alone. Name the Act and section only when a user naturally '
    'would. Never rely on outside legal knowledge, and never invent '
    'penalties, section numbers or dates.')


def personas():
  from ragas.testset.persona import Persona
  return [
      Persona(name='Practising advocate',
              role_description='An Indian advocate who needs the exact '
              'provision, its conditions and its status quickly. Precise, '
              'uses legal terms.'),
      Persona(name='Law student',
              role_description='A law student researching what a provision '
              'means and how it connects to other provisions. Asks "why" and '
              '"how" questions.'),
      Persona(name='Small business owner',
              role_description='A non-lawyer who runs a small business and '
              'asks plain-language questions about duties, penalties and '
              'deadlines. No legal jargon.'),
      Persona(name='Compliance officer',
              role_description='A compliance officer at a company or NBFC who '
              'asks what an Act or regulation requires and who it applies to.'),
      Persona(name='Citizen',
              role_description='An ordinary citizen asking what their rights '
              'or obligations are under a state or central law. Vague, '
              'conversational wording.'),
  ]


def build_generator(model: str):
  from langchain_google_genai import ChatGoogleGenerativeAI
  from ragas.llms import LangchainLLMWrapper
  from ragas.testset.synthesizers.generate import TestsetGenerator

  from .voyage_embeddings import VoyageRagasEmbeddings

  llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=model, temperature=0))
  return TestsetGenerator(
      llm=llm, embedding_model=VoyageRagasEmbeddings(),
      persona_list=personas(), llm_context=LLM_CONTEXT)


def query_distribution(llm):
  from ragas.testset.synthesizers.multi_hop import (
      MultiHopAbstractQuerySynthesizer, MultiHopSpecificQuerySynthesizer)
  from ragas.testset.synthesizers.single_hop.specific import (
      SingleHopSpecificQuerySynthesizer)
  kw = dict(llm=llm, llm_context=LLM_CONTEXT)
  return [
      (SingleHopSpecificQuerySynthesizer(**kw), 0.50),
      (MultiHopAbstractQuerySynthesizer(**kw), 0.25),
      (MultiHopSpecificQuerySynthesizer(**kw), 0.25),
  ]


def to_rows(testset, docs):
  """Ragas rows + gold labels recovered from the source provisions' metadata."""
  by_text = {d.page_content.strip(): d.metadata for d in docs}
  rows = []
  for i, s in enumerate(testset.to_list()):
    ctxs = s.get('reference_contexts') or []
    metas = [by_text.get(c.strip()) for c in ctxs]
    known = [m for m in metas if m]
    rows.append({
        'id': f'g{i:04d}',
        'user_input': s.get('user_input'),
        'reference': s.get('reference'),
        'reference_contexts': ctxs,
        'synthesizer_name': s.get('synthesizer_name'),
        'persona_name': s.get('persona_name'),
        'query_style': s.get('query_style'),
        'query_length': s.get('query_length'),
        'gold_chunk_ids': [m['chunk_id'] for m in known],
        'gold_act_ids': sorted({m['act_id'] for m in known}),
        'gold_sections': [
            {'act': m['title'], 'section_number': m['section_number'],
             'jurisdiction': m['jurisdiction'], 'state': m['state']}
            for m in known],
        'unmatched_contexts': len(ctxs) - len(known),
        'review_status': 'unreviewed',
    })
  return rows


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--size', type=int, default=5, help='questions to generate')
  ap.add_argument('--acts', type=int, default=40, help='Acts to sample')
  ap.add_argument('--chunks-per-act', type=int, default=8)
  ap.add_argument('--seed', type=int, default=7)
  ap.add_argument('--model', default=os.getenv('GEMINI_MODEL', 'gemini-3.8-flash'))
  ap.add_argument('--out', default='evals/golden/candidates.jsonl')
  ap.add_argument('--dry-run', action='store_true',
                  help='only sample from Qdrant and print stats; no LLM calls')
  args = ap.parse_args()

  docs, report = sample_chunks.sample_documents(
      n_acts=args.acts, chunks_per_act=args.chunks_per_act, seed=args.seed)
  print(f'Sampled {report.chunks} provisions from {report.acts} Acts '
        f'{report.per_stratum}')
  lens = sorted(len(d.page_content) for d in docs)
  if lens:
    print(f'chars min/median/max: {lens[0]}/{lens[len(lens)//2]}/{lens[-1]}')
  print('top Acts:', Counter(d.metadata['title'] for d in docs).most_common(5))
  if args.dry_run:
    return

  gen = build_generator(args.model)
  testset = gen.generate_with_chunks(
      docs, testset_size=args.size,
      query_distribution=query_distribution(gen.llm),
      raise_exceptions=False)
  rows = to_rows(testset, docs)

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open('w') as f:
    for r in rows:
      f.write(json.dumps(r, ensure_ascii=False) + '\n')
  print(f'Wrote {len(rows)} candidates to {out}')
  print('By type:', Counter(r['synthesizer_name'] for r in rows))
  print('Unmatched contexts:', sum(r['unmatched_contexts'] for r in rows))


if __name__ == '__main__':
  main()
