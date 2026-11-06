"""Run one question through the workflow; print the search plan and the answer.

Usage: uv run python test_query.py "your question"
"""

import asyncio
import sys

from google.adk.runners import InMemoryRunner
from google.genai import types

from agents.legal_agent.agent import root_agent

DEFAULT_Q = 'What is the punishment for cheating under the Bharatiya Nyaya Sanhita?'


def node_name(event):
  if not event.node_info:
    return None
  return event.node_info.path.split('/')[-1].split('@')[0]


async def main(question: str) -> None:
  runner = InMemoryRunner(app_name='legal_chat', agent=root_agent)
  session = await runner.session_service.create_session(
      app_name='legal_chat', user_id='tester')
  message = types.Content(
      role='user', parts=[types.Part.from_text(text=question)])
  print(f'Q: {question}\n')

  answer = ''
  final_nodes = {'deliver', 'withhold', 'no_answer', 'clarify', 'refuse',
                 'refuse_input'}
  async for event in runner.run_async(
      user_id='tester', session_id=session.id, new_message=message):
    texts = [p.text for p in (event.content.parts if event.content else [])
             if p.text]
    name = node_name(event)
    if name in ('input_guard', 'merge_sources') and texts:
      print(texts[0])  # input redaction report / "Searches: ... filters=..."
    elif name in final_nodes and texts:
      answer = f'[{name}]\n' + ''.join(texts)

  state = (await runner.session_service.get_session(
      app_name='legal_chat', user_id='tester', session_id=session.id)).state
  for note in state.get('plan', {}).get('notes', []):
    print(f'NOTE: {note}')
  print('safety:', state.get('safety_answer', {}).get('choice'),
        '| grounding P(true):', state.get('grounding_answer', {}).get('probability'))
  print(f'\nA: {answer}')


if __name__ == '__main__':
  asyncio.run(main(' '.join(sys.argv[1:]) or DEFAULT_Q))
