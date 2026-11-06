"""The workflow must work with a persistent (SQLite) session store."""

from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from agents.legal_agent.agent import build_workflow
from tests.test_workflow import FakeLlm, clf, hit, plan


async def test_workflow_state_survives_a_sqlite_session_store(tmp_path):
  db = f'sqlite+aiosqlite:///{tmp_path}/sessions.db'
  wf = build_workflow(
      planner_model=FakeLlm(responses=[plan({'query': 'x'}), plan({'query': 'y'})]),
      answer_model=FakeLlm(responses=['Answer [S1].', 'Answer [S1].']),
      classifier_model=FakeLlm(responses=clf()._responses * 2),
      search=lambda *a: [hit(1)])
  svc = DatabaseSessionService(db_url=db)
  runner = Runner(app_name='t', agent=wf, session_service=svc)
  session = await svc.create_session(app_name='t', user_id='u')
  for q in ('first question', 'second question'):
    msg = types.Content(role='user', parts=[types.Part(text=q)])
    _ = [e async for e in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg)]

  # Re-open the store from scratch, as a new process would.
  fresh = DatabaseSessionService(db_url=db)
  got = await fresh.get_session(app_name='t', user_id='u', session_id=session.id)
  assert got.state['question'] == 'second question'
  assert got.state['prev_question'] == 'first question'
  assert got.state['verification']['ok'] is True
  assert got.state['grounding_answer']['probability'] == 0.95
  assert got.state['safety_answer']['choice'] == 'ok'
  assert 'temp:sources' not in got.state
