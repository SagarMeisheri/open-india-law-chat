from agents.legal_agent import guardrails as g


def test_redacts_identifiers():
  text, rep = g.sanitize_input('mail raj@example.com or call +91 98765 43210')
  assert '[EMAIL]' in text and '[PHONE]' in text and 'raj@example.com' not in text
  assert rep['redacted'] == {'EMAIL_ADDRESS': 1, 'PHONE_NUMBER': 1}


def test_india_ids_use_checksum_and_shape():
  text, rep = g.sanitize_input(
      'aadhaar 2345 6789 0124, PAN ABCPE1234F, GSTIN 27AAPFU0939F1ZV')
  assert '[AADHAAR]' in text and '[PAN]' in text and '[GSTIN]' in text
  # Look-alikes (bad checksum, invalid PAN shape) are left alone.
  keep, rep = g.sanitize_input('aadhaar 1234 5678 9012 and WP ABCDE1234F of 2019')
  assert '1234 5678 9012' in keep and 'ABCDE1234F' in keep and not rep['redacted']


def test_legal_citations_are_never_masked():
  q = ('Kesavananda Bharati v. State of Kerala (1973) 4 SCC 225; Section 420 '
       'IPC read with section 34; Order 39 Rule 1 CPC; Act 45 of 1860; '
       '14 March 2019 in Mumbai')
  text, rep = g.sanitize_input(q)
  assert text == q and not rep['redacted']


def test_truncates_long_input():
  text, rep = g.sanitize_input('x' * 5000)
  assert len(text) == g.MAX_INPUT_CHARS and rep['truncated']


SOURCES = [{'fields': {'title': 'Negotiable Instruments Act',
                       'text': 'Where any cheque is dishonoured the drawer shall be deemed to have '
                               'committed an offence. Section 138.'}}]


def test_valid_answer_passes():
  ans = ('Dishonour of a cheque is an offence under Section 138 [S1]. It says '
         '"the drawer shall be deemed to have committed an offence" [S1].')
  r = g.verify_answer(ans, SOURCES)
  assert r['ok'], r


def test_fake_tag_quote_and_section_are_caught():
  ans = ('Under Section 420 [S3] "the accused shall be imprisoned for seven years always".')
  r = g.verify_answer(ans, SOURCES)
  text = ' '.join(r['warnings'])
  assert not r['ok']
  assert 'S3' in text and 'quoted text' in text and 'section 420' in text
