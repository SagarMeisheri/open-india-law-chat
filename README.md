# VaquillAI legal research chat

A research assistant for Indian legislation. You ask a question in plain English,
and it searches the [Open India Law](https://huggingface.co/datasets/vaquill/open-india-law)
corpus and answers **only from the provisions it retrieved**, with citations and
source links.

Built with [Google ADK](https://adk.dev) (a graph workflow of LLM and code
nodes), [Qdrant](https://qdrant.tech) (hybrid vector search), Voyage AI
(query embeddings) and Gemini (planning, classification, answering).

> **Status: working prototype.** Only half of one collection is loaded locally
> (see [Data coverage](#data-coverage)). Answers can be incomplete or wrong.
> **This is legal information, not legal advice**, and the data is a
> point-in-time snapshot, not current law. Always verify against the official
> source before relying on anything it says.

## How it works

```
START
  -> input_guard         Presidio redacts personal data before any API call
  -> safety_router       classifier: ok | injection | out_of_scope | fabrication
       |-- not ok -----> refuse
       `-- ok
  -> planner_agent       LLM: search | clarify | refuse; 1-4 sub-searches,
                         each with a query and metadata pre-filters
  -> search_one (xN)     filtered hybrid search (dense + BM25), in parallel,
                         with an unfiltered fallback
  -> merge_sources       de-duplicate and number sources [S1], [S2], ...
  -> sources_router      classifier: answerable | insufficient
       |-- insufficient -> "not found" message (no answer is generated)
       `-- answerable
  -> generate_answer     LLM writes a DRAFT the user cannot see yet
  -> grounding_router    classifier: is every claim supported by the sources?
       |-- yes ---------> deliver (answer + verification footer + disclaimer)
       `-- no ----------> withhold (show only the retrieved provisions)
```

Design choices worth knowing:

- **The answer is gated.** The draft is written inside a code node so it is
  never shown unless the grounding classifier approves it. The cost is that
  answers arrive all at once instead of streaming.
- **Code checks run besides the classifiers.** `guardrails.py` verifies that the
  `[S#]` tags exist and that quoted text and section numbers appear in the
  sources. Classifiers are LLM judgments and can be wrong; these checks cannot
  be talked out of it.
- **Filters are applied before the search**, so they narrow both the dense and
  the keyword search. If a filtered search finds nothing, it retries without
  filters and says so.
- **A failed search is not "nothing found".** The two cases give different
  messages.
- **Prompts live in `prompts.py`.** The classifier labels there are also the
  workflow routes; rename one and you must change the edges in `agent.py`.

## Requirements

- Docker (Qdrant runs in a container)
- [uv](https://docs.astral.sh/uv/) (Python 3.11 to 3.13 is managed for you)
- A [Google AI Studio](https://aistudio.google.com/app/apikey) key and a
  [Voyage AI](https://dash.voyageai.com) key
- Disk: about **6 GB** for one `acts_india` shard, plus **~11 GB free at peak**
  while it restores and optimizes. The full corpus is far larger (see below).

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Configure keys
cp .env.example .env          # then fill in GOOGLE_API_KEY and VOYAGE_API_KEY

# 3. Start Qdrant (v1.16.3, bound to localhost only)
docker compose up -d

# 4. Load data. Shard 1 of acts_india is 5.3 GB; shard 0 is 5.9 GB.
SHARD_LIST=1 MIN_FREE_GB=12 scripts/restore_collection.sh acts_india 2
#    all of acts_india (11.2 GB download, ~35 GB free recommended):
#    scripts/restore_collection.sh acts_india 2

# 5. Check what loaded (field names, status values, sparse-vector check)
uv run python scripts/probe_collection.py acts_india

# 6. Ask a question
uv run python test_query.py "What is the punishment for dishonour of a cheque under the Negotiable Instruments Act?"
```

The restore takes roughly 15 minutes per shard and Qdrant keeps optimizing for
several minutes afterwards (`curl -s localhost:6333/collections/acts_india | jq .result.status`
goes from `yellow` to `green`).

### Chat UI

```bash
uv run adk web agents --port 8001 \
  --session_service_uri "sqlite+aiosqlite:///$(pwd)/sessions.db"
```

Open <http://localhost:8001> and pick `legal_agent`. Port 8000 is `adk web`'s
default; use another port if something else already holds it. Chats persist in
`sessions.db` (git-ignored). The trace view shows each node, the search filters
and the classifier decisions. `adk web` is a **development** UI; build your own
front end against `adk api_server` for anything user-facing.

### Tests

```bash
uv run pytest -q
```

The suite runs offline with scripted fake LLMs and a fake search, with no keys
or Qdrant needed. It checks routing, gating, filters, PII redaction, session
persistence and the guardrails. **It cannot tell you whether the real models
behave well**: for that, run `test_query.py` with real keys. (One real-model
bug, an empty-string enum that Gemini rejects, was only found that way and now
has its own test.)

## Configuration

All settings are environment variables (`.env`). See `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `GOOGLE_API_KEY` | required | Gemini access |
| `VOYAGE_API_KEY` | required | Embeds the user's question |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Planner and answer model |
| `CLASSIFIER_MODEL` | `GEMINI_MODEL` | The three classifier calls |
| `VOYAGE_MODEL` | `voyage-4` | Query embedding model (see note) |
| `QDRANT_URL` | `http://localhost:6333` | |
| `QDRANT_COLLECTION` | `acts_india` | |
| `QDRANT_TIMEOUT` | `30` | Seconds |
| `SPARSE_MODE` | `bm25` | `off` for dense-only |
| `GROUNDING_THRESHOLD` | `0.7` | Minimum P(grounded) to show an answer |

**Embedding model note.** The stored vectors were made with Voyage's `voyage-4`
series, so queries must use a model from that series: `voyage-4`,
`voyage-4-lite`, `voyage-4-large` and the open-weight `voyage-4-nano` are
documented as compatible with each other. Other models (including
`voyage-law-2`, OpenAI, bge-m3) produce vectors in a different space and give
meaningless scores. `voyage-4-nano` outputs 2048 dimensions by default; it must
be truncated to 1024 to match. Whether a smaller query model retrieves as well
as `voyage-4` on this data is untested.

## Project layout

```
agents/legal_agent/
  agent.py        the workflow graph (nodes and edges)
  prompts.py      every prompt, classifier question and user-facing message
  retrieval.py    Qdrant hybrid search, filters, state-name -> slug mapping
  guardrails.py   Presidio PII redaction and citation verification
scripts/
  restore_collection.sh   create a collection and restore snapshot shards
  probe_collection.py     inspect payload fields, values and sparse vectors
  inspect_collection.py   status plus one sample payload
tests/                    offline tests (fake LLMs, fake search)
test_query.py             run one question end to end
docker-compose.yml        Qdrant v1.16.3, localhost only
```

## Data coverage

The data is the [Open India Law](https://huggingface.co/datasets/vaquill/open-india-law)
corpus. This project uses only the `acts_india` collection (Acts and
regulator instruments, 1,098,577 provisions in 2 shards).

- **Loaded here: shard 1 only, 518,226 provisions (~47%).** Some Acts and
  sections are therefore missing and the assistant may say it cannot find them.
  Restore shard 0 for full coverage.
- **Not included:** court judgments (`legal_corpus_v1`, 273 GB) and tribunal
  decisions (`legal_corpus_v2`, 180 GB). Judgment search would need much more
  disk and a different retrieval design.
- **No citation graph.** The dataset cannot tell whether a provision was later
  amended or a judgment overruled; do not treat output as current law.

Facts learned from the real data (they differ from the dataset README):

- `state` holds lowercase slugs (`madhya-pradesh`, `jammu-kashmir`,
  `andaman-nicobar`) and is `central` for **both** central Acts and regulator
  instruments. Use `jurisdiction` (`central` / `state` / `regulatory`) and
  `department` (`rbi`, `sebi`, ...) to tell them apart.
- `act_status`: `in_force`, `superseded`, `repealed`, `spent`.
  `section_status`: `in_force`, `omitted`, `repealed`, `substituted`.
- `section_number` and `title` are indexed, so an exact "section 138 of the NI
  Act" lookup can be a filter instead of a fuzzy search.

## Known limitations

- **Keyword (BM25) search covers about half the chunks.** About 53% of stored
  sparse vectors match FastEmbed's BM25; the rest use a raw-count encoding with
  a hash I could not identify, so keyword queries never match them. Dense
  search covers everything. The collection also has no IDF weighting (enabling
  Qdrant's IDF modifier improved one exact-section query; it is reverted to the
  published config).
- **Classifier probabilities are the model's stated values**, not measured
  frequencies. The grounding judge scored 100% on every real answer so far, so
  the 0.7 threshold has not really been tested. Build an evaluation set,
  including deliberately wrong drafts, before trusting it.
- **Latency is about 30 seconds per answer** (six or so model calls, plus
  retrieval).
- A fictional Act in a question is currently refused as "out of scope" rather
  than answered with "not found".
- `google.adk.labs.classifiers` is an experimental ADK package and may change.

## Privacy and safety

- Questions are sent to **Voyage AI** (embedding) and **Google** (Gemini). Before
  that, Presidio masks email, phone, Aadhaar, PAN, GSTIN, passport, voter ID,
  card numbers and IP addresses. It deliberately does **not** mask names,
  places or dates (that would break queries like "Kesavananda Bharati v. State
  of Kerala"), so free-text personal details still leave the machine.
- `sessions.db` stores chat history. It is git-ignored; treat it as sensitive.
- Qdrant has no authentication and is bound to `127.0.0.1` only. Do not expose
  it without an API key.
- Do not paste privileged or confidential client material into this tool.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `address already in use` on port 8000 | Another `adk web` is running. Use `--port 8001`. |
| `400 INVALID_ARGUMENT ... enum ... cannot be empty` | An enum with `''` in a structured-output schema; Gemini rejects it. Use a named value such as `'any'`. |
| "The search itself failed ... timed out" | Qdrant is busy optimizing after a restore. Wait for `green`, or raise `QDRANT_TIMEOUT`. |
| Restore refuses to start (`Not enough free disk space`) | Free space, or lower `MIN_FREE_GB` (at your own risk: a full disk can break other Docker databases). |
| Model not found | Set `GEMINI_MODEL` to an ID your key lists (`client.models.list()`). |
| Filters match nothing | Check values with `scripts/probe_collection.py`; `state` uses slugs. |
| Qdrant log: `Can't notify optimizers... Restart is required` | Seen once and harmless in testing (the collection kept optimizing and went `green`). If a collection stays `yellow` with idle CPU, run `docker restart qdrant`. |

## Licence and attribution

- **This project's code:** [Apache-2.0](LICENSE) (the same licence the dataset
  uses for its own scripts). Copyright and third-party notices are in
  [NOTICE](NOTICE).
- **The data:** the [Open India Law](https://huggingface.co/datasets/vaquill/open-india-law)
  dataset by Vaquill AI, licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). **Attribution is
  required**, including in any product built on it. This repo does not contain
  or redistribute the data; it restores the authors' published snapshots
  unchanged into a local Qdrant. Every answer the app delivers (and every
  withheld-answer notice) ends with the credit line in `ATTRIBUTION`
  (`agents/legal_agent/prompts.py`). Keep it if you fork or deploy this. Some
  sources (eCourts, NCLT, NCLAT) require their material to be prominently
  acknowledged; the credit also serves that purpose.
- **The legal text itself:** a government work (Copyright Act 1957, s.17(d);
  reproducible under s.52(1)(q)).
- **Not endorsed:** this project is not affiliated with Vaquill AI or any
  Indian court, regulator or government body.
- The dataset's README notes that two regulators (IRDAI, CPCB) were scraped
  against their `robots.txt`; consider that before commercial use of those
  sources.
