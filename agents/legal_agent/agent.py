"""Multi-step legal research workflow (Google ADK 2.x) with classifier routers.

  START -> input_guard -> safety_router --ok--> planner_agent -> route_plan
                              |                                    |
                              +-other--> refuse_input              +-search--> search_one (xN, parallel)
                                                                   +-clarify / refuse         |
                                                                                       merge_sources
                                                                                              |
     +-----insufficient--> no_answer <-----------------------------------------------sources_router
     |                                                                                        | sufficient / partial
     |                       deliver <--True-- grounding_router <-- generate_answer <---------+
     |                       withhold <-False--/

Classifier routers (ClassifierNode + LlmClassifier, each one cheap LLM call):
* safety_router     Choice: ok | injection | out_of_scope | fabrication
* sources_router    Choice: answerable | insufficient  (do the sources answer any part?)
* grounding_router  Binary: is every claim in the DRAFT supported by the sources?

Code nodes: input_guard (Presidio PII redaction), search_one (filtered hybrid
search), merge_sources, generate_answer, deliver/withhold/no_answer/refuse.

The draft answer is produced inside `generate_answer` (a function node calling
the model directly) so the user never sees it unless grounding_router approves.
The cost is that the answer arrives all at once instead of streaming.

State: 'question', 'prev_question', 'plan', 'safety_answer', 'grounding_answer',
'verification' persist with the session. Retrieved source text is
'temp:sources' (this run only, never stored).
"""

import os
from pathlib import Path
from typing import Callable, Literal

from dotenv import load_dotenv
from google.adk import Context, Event, Workflow
from google.adk.agents import LlmAgent
from google.adk.labs.classifiers import Binary, Choice, ClassifierNode, LlmClassifier
from google.adk.models.llm_request import LlmRequest
from google.adk.models.registry import LLMRegistry
from google.adk.workflow import DEFAULT_ROUTE, node
from google.genai import types
from pydantic import BaseModel, Field

from . import guardrails, prompts, retrieval

# Load the project-root .env (ADK would otherwise only look beside this file).
load_dotenv(Path(__file__).resolve().parents[2] / '.env')

LLM_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.8-flash')
# Classifiers are short structured calls; a cheaper model is fine. Unverified
# model id: set CLASSIFIER_MODEL if you want a lighter one than LLM_MODEL.
CLASSIFIER_MODEL = os.getenv('CLASSIFIER_MODEL', LLM_MODEL)
# P(true) the grounding classifier must reach before an answer is shown.
GROUNDING_THRESHOLD = float(os.getenv('GROUNDING_THRESHOLD', '0.7'))
MAX_SEARCHES = 4
TOP_K_PER_SEARCH = 6
MAX_SOURCES = 10


class SearchPlan(BaseModel):
  """One sub-search: a query plus metadata pre-filters."""

  query: str = Field(
      description=(
          'Compact search query. Keep exact identifiers verbatim (section'
          ' numbers, Act names, years).'))
  purpose: str = Field(
      default='', description='Which part of the question this search answers.')
  jurisdictions: list[str] = Field(
      default_factory=list,
      description="'central' and/or state/UT names. Empty list = no restriction.")
  status: Literal['', 'in_force', 'repealed', 'spent', 'superseded'] = Field(
      default='', description="Provision status filter; '' = any.")
  year_from: int = Field(default=0, description='Earliest Act year; 0 = none.')
  year_to: int = Field(default=0, description='Latest Act year; 0 = none.')


class Plan(BaseModel):
  """The planner's decision."""

  route: Literal['search', 'clarify', 'refuse']
  message: str = Field(
      default='',
      description=(
          "For 'clarify': the one question to ask the user. For 'refuse': a"
          " short reason. Empty for 'search'."))
  searches: list[SearchPlan] = Field(
      default_factory=list,
      description="1-4 sub-searches when route is 'search'; else empty.")


def _source_line(i: int, source: dict) -> str:
  f = source.get('fields', {})
  bits = [str(f[k]) for k in ('title', 'section_number', 'source_url') if f.get(k)]
  if not bits:
    bits = [str(v)[:120] for v in list(f.values())[:2]]
  return f'[S{i}] ' + ' | '.join(bits)


def build_workflow(
    planner_model=LLM_MODEL,
    answer_model=LLM_MODEL,
    classifier_model=CLASSIFIER_MODEL,
    search: Callable[[str, dict, int], list[dict]] = retrieval.hybrid_search,
) -> Workflow:
  """Builds the workflow. Models (name or BaseLlm) and `search` are injectable."""
  classifier = LlmClassifier(model=classifier_model)
  answer_llm = answer_model if not isinstance(answer_model, str) else None

  def input_guard(node_input: str, question: str = ''):
    clean, report = guardrails.sanitize_input(node_input)
    state = {'question': clean, 'prev_question': question}
    if report['truncated'] or report['redacted']:
      yield Event(message=f'Input check: {report}')
    yield Event(output=clean, state=state)

  safety_router = ClassifierNode(
      name='safety_router',
      classifier=classifier,
      question=Choice(
          instructions=prompts.SAFETY_INSTRUCTIONS,
          criteria=prompts.SAFETY_CRITERIA),
      answer_key='safety_answer',
  )

  def refuse_input(ctx: Context, node_input: str):
    label = ctx.state.get('safety_answer', {}).get(
        'choice', prompts.DEFAULT_REFUSAL_LABEL)
    msg = prompts.REFUSALS.get(
        label, prompts.REFUSALS[prompts.DEFAULT_REFUSAL_LABEL])
    yield Event(message=msg)
    yield Event(output=msg)

  planner_agent = LlmAgent(
      name='planner_agent',
      model=planner_model,
      description='Plans the research: search, clarify or refuse.',
      instruction=prompts.PLANNER_INSTRUCTION,
      output_schema=Plan,
  )

  def route_plan(node_input: Plan):
    plan = node_input
    if plan.route == 'search' and plan.searches:
      searches = [s.model_dump() for s in plan.searches[:MAX_SEARCHES]]
      return Event(output=searches, route='search')
    if plan.route == 'refuse':
      return Event(output=plan.message or prompts.PLANNER_REFUSE_DEFAULT,
                   route='refuse')
    return Event(
        output=plan.message or prompts.PLANNER_CLARIFY_DEFAULT,
        route='clarify')

  def search_one_fn(node_input: SearchPlan, question: str):
    plan = node_input
    notes: list[str] = []
    jurisdictions, rejected = retrieval.normalize_jurisdictions(
        plan.jurisdictions)
    if rejected:
      notes.append(f'Ignored unknown jurisdictions: {rejected}.')
    filters = {k: v for k, v in {
        'jurisdictions': jurisdictions, 'status': plan.status,
        'year_from': plan.year_from, 'year_to': plan.year_to}.items() if v}
    query = plan.query.strip() or question
    try:
      hits = search(query, filters, TOP_K_PER_SEARCH)
      if not hits and filters:
        notes.append(f'No results for "{query}" with filters {filters}; '
                     'searched again WITHOUT filters.')
        hits = search(query, {}, TOP_K_PER_SEARCH)
        filters = {}
    except Exception as exc:  # report, never invent results
      notes.append(f'Search for "{query}" failed: {type(exc).__name__}: {exc}')
      hits = []
    return {'query': query, 'purpose': plan.purpose, 'filters': filters,
            'notes': notes, 'hits': hits}

  search_one = node(search_one_fn, name='search_one', parallel_worker=True)

  def merge_sources(node_input: list[dict], question: str):
    # Interleave by rank so every sub-search is represented, de-duplicate by id.
    sources, seen = [], set()
    depth = max((len(r['hits']) for r in node_input), default=0)
    for rank in range(depth):
      for r in node_input:
        if rank < len(r['hits']):
          h = r['hits'][rank]
          if h['id'] not in seen and len(sources) < MAX_SOURCES:
            seen.add(h['id'])
            sources.append({**h, 'retrieved_for': r['query']})

    searches = [{'query': r['query'], 'purpose': r['purpose'],
                 'filters': r['filters']} for r in node_input]
    notes = [n for r in node_input for n in r['notes']]
    blocks = []
    for i, s in enumerate(sources, start=1):
      fields = '\n'.join(f'  {k}: {v}' for k, v in s['fields'].items())
      blocks.append(f'[S{i}] (retrieved for: {s["retrieved_for"]})\n{fields}')
    context = (
        f'QUESTION: {question}\n\n'
        'SEARCHES RUN:\n' + '\n'.join(
            f'- {x["query"]} | purpose: {x["purpose"] or "-"} | filters: {x["filters"] or "none"}'
            for x in searches) +
        f'\n\nNOTES: {" ".join(notes) or "none"}\n\n'
        f'SOURCES:\n{chr(10).join(blocks) if blocks else "NO SOURCES FOUND."}'
    )
    yield Event(message='Searches: ' + '; '.join(
        f'{x["query"]!r} filters={x["filters"] or "none"}' for x in searches))
    yield Event(
        output=context,
        state={'plan': {'searches': searches, 'notes': notes},
               'temp:sources': sources})

  sources_router = ClassifierNode(
      name='sources_router',
      classifier=classifier,
      question=Choice(
          instructions=prompts.SOURCES_INSTRUCTIONS,
          criteria=prompts.SOURCES_CRITERIA),
  )

  def no_answer(ctx: Context, node_input: str):
    searches = ctx.state.get('plan', {}).get('searches', [])
    ran = '; '.join(f'"{x["query"]}"' for x in searches) or 'none'
    msg = prompts.NO_ANSWER_TEMPLATE.format(searches=ran)
    yield Event(message=msg)
    yield Event(output=msg)

  async def generate_answer(ctx: Context, node_input: str):
    """Writes the DRAFT answer without emitting it to the user."""
    nonlocal answer_llm
    if answer_llm is None:
      answer_llm = LLMRegistry.new_llm(answer_model)
    request = LlmRequest(
        model=answer_llm.model,
        contents=[types.Content(role='user', parts=[types.Part(text=node_input)])],
        config=types.GenerateContentConfig(system_instruction=prompts.ANSWER_INSTRUCTION),
    )
    draft = ''
    async for response in answer_llm.generate_content_async(request, stream=False):
      if response.content:
        draft += ''.join(p.text for p in response.content.parts
                         if p.text and not p.thought)
    check = guardrails.verify_answer(draft, ctx.state.get('temp:sources', []))
    return Event(
        output={'question_and_sources': node_input, 'draft_answer': draft},
        state={'verification': check})

  grounding_router = ClassifierNode(
      name='grounding_router',
      classifier=classifier,
      question=Binary(
          instructions=prompts.GROUNDING_INSTRUCTIONS,
          criteria=prompts.GROUNDING_CRITERIA),
      threshold=GROUNDING_THRESHOLD,
      answer_key='grounding_answer',
  )

  def deliver(ctx: Context, node_input: dict):
    draft = node_input['draft_answer']
    check = ctx.state.get('verification', {'ok': True, 'warnings': [], 'tags': 0})
    p = ctx.state.get('grounding_answer', {}).get('probability')
    note = f' Grounding check: {p:.0%} confident.' if p is not None else ''
    yield Event(message=f'{draft}\n\n{guardrails.format_footer(check)}{note}')
    yield Event(output=draft)

  def withhold(ctx: Context, node_input: dict):
    sources = ctx.state.get('temp:sources', [])
    p = ctx.state.get('grounding_answer', {}).get('probability')
    lines = '\n'.join(_source_line(i, s) for i, s in enumerate(sources, start=1))
    msg = prompts.WITHHOLD_TEMPLATE.format(
        probability=f'{p:.0%}' if p is not None else 'low',
        sources=lines, disclaimer=prompts.DISCLAIMER)
    yield Event(message=msg)
    yield Event(output=msg)

  def clarify(node_input: str):
    yield Event(message=node_input)
    yield Event(output=node_input)

  def refuse(node_input: str):
    msg = prompts.PLANNER_REFUSE_TEMPLATE.format(reason=node_input)
    yield Event(message=msg)
    yield Event(output=msg)

  return Workflow(
      name='legal_research_workflow',
      input_schema=str,
      edges=[
          ('START', input_guard, safety_router),
          (safety_router, {'ok': planner_agent, DEFAULT_ROUTE: refuse_input}),
          (planner_agent, route_plan),
          (route_plan, {
              'search': search_one, 'clarify': clarify, 'refuse': refuse}),
          (search_one, merge_sources, sources_router),
          (sources_router, {
              'answerable': generate_answer, 'insufficient': no_answer}),
          (generate_answer, grounding_router),
          (grounding_router, {True: deliver, False: withhold}),
      ],
  )


root_agent = build_workflow()
