"""Deterministic guardrails (no LLM): input redaction and output verification.

Why code instead of prompts: prompt rules can be ignored by the model, while
these checks always run. They complement, not replace, the prompt rules.
"""

import re

from .prompts import DISCLAIMER, LEGAL_NOTICE  # DISCLAIMER re-exported for callers

MAX_INPUT_CHARS = 4000

# --- Input: redact personal data before it reaches Gemini / Voyage -----------
# Uses Microsoft Presidio. Deliberately limited to identifier entities: masking
# PERSON / LOCATION / ORGANIZATION / DATE_TIME would destroy legal queries such
# as "Kesavananda Bharati v. State of Kerala". The India recognizers are not
# enabled in Presidio by default, so they are registered here. IN_AADHAAR is
# checksum-validated and IN_PAN needs a valid shape, which keeps case numbers
# like "WP ABCDE1234F" from being masked.
REDACT_ENTITIES = {
    'EMAIL_ADDRESS': '[EMAIL]',
    'PHONE_NUMBER': '[PHONE]',
    'IN_AADHAAR': '[AADHAAR]',
    'IN_PAN': '[PAN]',
    'IN_PASSPORT': '[PASSPORT]',
    'IN_VOTER': '[VOTER_ID]',
    'IN_GSTIN': '[GSTIN]',
    'CREDIT_CARD': '[CARD]',
    'IBAN_CODE': '[IBAN]',
    'IP_ADDRESS': '[IP]',
}
SCORE_THRESHOLD = 0.4
SPACY_MODEL = 'en_core_web_sm'  # small (~12 MB); the default would be ~400 MB

_analyzer = None
_anonymizer = None


def _engines():
  """Builds the Presidio engines once (loading spaCy takes about a second)."""
  global _analyzer, _anonymizer
  if _analyzer is None:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer import predefined_recognizers as pr
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    nlp = NlpEngineProvider(nlp_configuration={
        'nlp_engine_name': 'spacy',
        'models': [{'lang_code': 'en', 'model_name': SPACY_MODEL}],
    }).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=['en'])
    for name in ('InAadhaarRecognizer', 'InPanRecognizer',
                 'InPassportRecognizer', 'InVoterRecognizer',
                 'InGstinRecognizer'):
      analyzer.registry.add_recognizer(getattr(pr, name)())
    _analyzer, _anonymizer = analyzer, AnonymizerEngine()
  return _analyzer, _anonymizer


def sanitize_input(text: str) -> tuple[str, dict]:
  """Truncates and redacts PII. Returns (clean_text, report)."""
  from presidio_anonymizer.entities import OperatorConfig

  report = {'truncated': False, 'redacted': {}}
  if len(text) > MAX_INPUT_CHARS:
    text = text[:MAX_INPUT_CHARS]
    report['truncated'] = True

  analyzer, anonymizer = _engines()
  found = analyzer.analyze(
      text=text, language='en', entities=list(REDACT_ENTITIES),
      score_threshold=SCORE_THRESHOLD)
  if not found:
    return text, report

  result = anonymizer.anonymize(
      text=text, analyzer_results=found,
      operators={e: OperatorConfig('replace', {'new_value': tag})
                 for e, tag in REDACT_ENTITIES.items()})
  for item in result.items:
    report['redacted'][item.entity_type] = (
        report['redacted'].get(item.entity_type, 0) + 1)
  return result.text, report


# --- Output: check the answer against the sources it was given ---------------
# Matches [S2] and also grouped forms such as [S3, S4] or [S1; S2].
_TAG_GROUP = re.compile(r'\[(S\d+(?:\s*[,;]\s*S?\d+)*)\]')
_QUOTE = re.compile(r'["“]([^"”]{25,}?)["”]')
_SECTION = re.compile(r'\b[Ss]ections?\s+(\d+[A-Za-z]?)\b')


def _norm(s: str) -> str:
  return re.sub(r'\s+', ' ', re.sub(r'[‘’“”"\']', '', s)).strip().lower()


def verify_answer(answer: str, sources: list[dict]) -> dict:
  """Checks tags, quotes and section numbers in `answer` against `sources`.

  Returns {'ok': bool, 'warnings': [...], 'tags': n}. It cannot prove the
  answer is *right*, only that what it cites actually exists in the sources.
  """
  warnings: list[str] = []
  corpus = _norm(' '.join(
      str(v) for s in sources for v in s.get('fields', {}).values()))

  tags = {int(n) for grp in _TAG_GROUP.findall(answer)
          for n in re.findall(r'\d+', grp)}
  bad = sorted(t for t in tags if not 1 <= t <= len(sources))
  if bad:
    warnings.append(f'cites sources that do not exist: {["S%d" % t for t in bad]}')

  if sources and not tags and len(answer) > 300:
    warnings.append('long answer with no [S#] citations')

  for quote in _QUOTE.findall(answer):
    if _norm(quote) not in corpus:
      warnings.append(f'quoted text not found in any source: "{quote[:60]}..."')

  for sec in dict.fromkeys(_SECTION.findall(answer)):
    if sources and not re.search(rf'(?<![\w]){re.escape(sec.lower())}(?![\w])', corpus):
      warnings.append(f'section {sec} is mentioned but appears in no source')

  return {'ok': not warnings, 'warnings': warnings, 'tags': len(tags)}


def format_footer(result: dict) -> str:
  if result['ok']:
    head = f'Check passed: {result["tags"]} source tag(s) valid; quotes and section numbers appear in the sources.'
  else:
    head = 'CHECK FAILED, do not rely on these points without verifying:\n' + '\n'.join(
        f'- {w}' for w in result['warnings'])
  return f'{head}\n\n{LEGAL_NOTICE}'
