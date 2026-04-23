# RxIntel — Drug Intelligence Agent

Clinician-facing Q&A over DrugBank. Combines a Neo4j knowledge graph
(drug-drug interactions, drug-enzyme relationships) with a ChromaDB
vector store (descriptions, indications, mechanisms, pharmacodynamics,
contraindications), orchestrated by a LangGraph multi-agent pipeline
that classifies intent, dispatches to the right retriever(s), reasons
over the evidence, and runs every answer through a critic before it
reaches the user.

## What it does

Five modes of questioning, auto-detected from the query:

| Mode | Example query | Retrievers |
|------|---------------|------------|
| `ddi_check` | "Is warfarin safe with aspirin?" | Graph (INTERACTS\_WITH + VIA\_ENZYME multi-hop) |
| `alternatives` | "Alternatives to ibuprofen for a CKD patient" | Vector (indications + contraindications) |
| `describe` | "What is semaglutide and how does it work?" | Vector (description + mechanism + PD) |
| `hybrid` | "Which CYP3A4 inhibitors interact with atorvastatin?" | Graph + vector, fused via RRF |
| `polypharmacy` | "Patient on warfarin, metformin, atorvastatin, omeprazole, clopidogrel — any concerns?" | Graph (all-pairs INTERACTS\_WITH) |

Every answer is scored by a critic agent (factual accuracy, safety,
completeness). Composite < 0.75 triggers a retry with revision
feedback; two failed retries escalate to "human review" with a red
banner in the UI.

## Architecture

```
 query
   │
   ▼
 entity_resolver  (rapidfuzz → Med7 → FAISS fallback)
   │
   ▼
 query_router    (Llama-3.3-70B, JSON-only, temperature 0)
   │
   ▼
 retrieval_dispatch
   │
   ├─► graph_retriever   (Neo4j, action-compatibility filter)
   └─► vector_retriever  (ChromaDB, 5 collections, MiniLM embeddings)
           │
           ▼
       RRF fusion (k=60) for hybrid mode
           │
           ▼
 reasoning_agent (mode-aware prompts, evidence-grounded)
           │
           ▼
 critic_agent   (retry up to 2× on rejection, then escalate)
           │
           ▼
       final_output
```

Four services run under Docker Compose: **api** (FastAPI),
**streamlit** (UI), **neo4j** (graph DB), **mlflow** (traces).
ChromaDB and the gazetteer pickle are host-mounted artifacts produced
by the ETL step.

## Prerequisites

- Docker Desktop (tested on Apple Silicon — `docker-compose.yml`
  pins `platform: linux/amd64` for the `api` and `streamlit` images)
- Python 3.11+ (for one-time ETL and local development)
- A [Groq API key](https://console.groq.com/keys) — free tier works,
  daily limit ~100k tokens
- **A DrugBank XML dump** (`full_database.xml`, 2.4 GB). DrugBank's
  academic license prohibits redistribution; you must download it
  yourself from https://go.drugbank.com/releases/latest. Place it at
  `data/full_database.xml`.

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/phanich004/RxIntel.git
cd RxIntel
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Copy the env template and fill in secrets:

```bash
cp .env.example .env    # if an .env.example is not present, create .env
```

Minimum variables (`.env`):

```bash
# --- Neo4j (local container) ---
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<pick-a-password>
NEO4J_DATABASE=neo4j

# --- Groq ---
GROQ_API_KEY=gsk_...
ROUTER_MODEL=llama-3.3-70b-versatile
REASONING_MODEL=llama-3.3-70b-versatile
CRITIC_MODEL=llama-3.3-70b-versatile
GROQ_MIN_DELAY_MS=200

# --- Agent tuning ---
MAX_RETRIES=3
CRITIC_APPROVAL_THRESHOLD=0.75
INSUFFICIENT_EVIDENCE_COSINE=0.60

# --- Artifacts ---
CHROMA_PERSIST_DIR=./chroma_db
GAZETTEER_PATH=./gazetteer.pkl

# --- MLflow ---
MLFLOW_TRACKING_URI=http://localhost:5000

# --- Demo mode (short-circuit to pre-seeded responses) ---
DEMO_MODE=false
```

### 2. Build the ETL artifacts

```bash
# Start Neo4j (takes ~10s)
docker compose up -d neo4j

# Parse DrugBank XML and load Neo4j (~10 min, ~11k drugs + 2.3M edges).
python -m etl.parse_drugbank \
    --xml data/full_database.xml \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password "$NEO4J_PASSWORD"

# Build the ChromaDB vector store (~20 min, ~28k chunks, ~160 MB on disk).
python -m etl.build_vector_store --xml data/full_database.xml

# Build the rapidfuzz gazetteer (~2 min, ~45k strings).
python -m etl.build_gazetteer --xml data/full_database.xml
```

Expected artifacts after this step:

```
./chroma_db/        # vector store, ~160 MB
./gazetteer.pkl     # drug-name lookup, ~3 MB
./gazetteer_fuzzy.pkl
Neo4j:              # ~11,438 nodes, ~2,326,114 INTERACTS_WITH edges
                    #                ~5,856 VIA_ENZYME edges
```

### 3. Launch the stack

```bash
docker compose up -d
```

Services (host ports):

| Service | URL | Purpose |
|---------|-----|---------|
| Streamlit UI | http://localhost:8501 | Clinician interface |
| FastAPI | http://localhost:8000 | Programmatic `/ask` + `/health` |
| FastAPI docs | http://localhost:8000/docs | Swagger |
| MLflow | http://localhost:5001 | Traces (remapped from 5000 to avoid macOS AirPlay) |
| Neo4j Browser | http://localhost:7474 | Graph exploration |

First request warms the embedding model (~5s). Subsequent queries land
in ~3–8 s end-to-end depending on mode and retry count.

### Optional: demo mode

Pre-seeded canned responses for the 4 example-button queries, no
Groq tokens consumed, simulated 3.5s latency. Handy for reliable
demos.

```bash
python -m scripts.seed_demo        # one-time: refresh canned responses
# edit .env -> DEMO_MODE=true
docker compose build api           # bake demo_responses.json into image
docker compose up -d
```

## Using it

### Streamlit UI

Open http://localhost:8501, type a question or click one of the four
example buttons. The page progressively discloses:

- **Recommendation** — the headline a clinician reads first
- **Mode + severity badges** (or an escalation banner if the critic
  gave up after 2 retries)
- **Mechanism** — PK/PD reasoning
- **Interacting pairs / Candidates** — per-pair severity rationale
- **Sources** — linked DrugBank IDs
- **Raw evidence** — every graph row and vector chunk the pipeline saw
- **Critic scores** — the three rubrics plus composite
- **Footer** — latency, retry count, approval status
- **Copy JSON** — download the full response envelope for tickets or
  logs
  
### API

```bash
curl -s http://localhost:8000/ask \
     -H 'Content-Type: application/json' \
     -d '{"query": "Is warfarin safe with aspirin?"}' | jq .
```

Rate-limited to 10 req/min per IP.

### Example queries

Copy-paste into the UI:

- `Is warfarin safe with aspirin?` — ddi\_check, Moderate
- `Alternatives to ibuprofen for a patient with chronic kidney disease`
  — alternatives
- `What is semaglutide and how does it work?` — describe
- `Which CYP3A4 inhibitors interact with atorvastatin?` — hybrid
- `Patient is taking warfarin, metformin, atorvastatin, omeprazole, and clopidogrel. Any concerns?`
  — polypharmacy, flags warfarin + clopidogrel as Major

## Tests

```bash
# full suite (mocks Neo4j/Groq where possible; requires NEO4J_PASSWORD
# in env for the handful of live graph tests).
pytest

# specific module
pytest tests/test_graph_retriever.py -v

# live end-to-end — remove the @pytest.mark.skip in
# tests/test_graph.py::test_live_end_to_end_warfarin_aspirin first.
pytest tests/test_graph.py::test_live_end_to_end_warfarin_aspirin -v
```

Live Groq tests (`tests/test_router.py`, critic/reasoning-agent live
cases) are skipped by default so a plain `pytest` run never burns
daily quota.

Type check:

```bash
mypy --strict agent/ api/ etl/
```

## Project layout

```
agent/          LangGraph pipeline — nodes, prompts, schemas, graph wiring
api/            FastAPI service — /ask, /health, rate limiting, DEMO_MODE
ui/             Streamlit app — components, styles, entry point
etl/            One-time DrugBank loaders (Neo4j, ChromaDB, gazetteer)
scripts/        Operational scripts (seed_demo.py)
tests/          pytest suite
data/           DrugBank XML lives here (gitignored); demo_responses.json
                is the one tracked artifact
docker-compose.yml   Four-service topology
Dockerfile.api       Multi-stage build, CPU-only torch
Dockerfile.streamlit Slim UI image
```

## Troubleshooting

- **`ports are not available: ... :5000`** — macOS AirPlay Receiver
  occupies port 5000. The compose file remaps MLflow to `5001:5000`
  on the host; if you still hit a collision, `lsof -nP -iTCP:5001
  -sTCP:LISTEN` to find and kill the squatter.
- **`groq.RateLimitError 429 ... tokens per day`** — you've hit
  Groq's free-tier 100k-token daily limit. Either wait (resets at
  midnight Pacific) or flip `DEMO_MODE=true` and rebuild the api
  image.
- **`service "neo4j" has no container to start`** — a previous
  `docker compose down` removed the container. Use
  `docker compose up -d neo4j` to recreate it; `start` only resumes
  an existing stopped container.
- **Empty graph / empty vector results** — the ETL step didn't
  finish. Check `MATCH (n) RETURN count(n)` in Neo4j Browser should
  return ~11,438; `ls chroma_db/` should be non-empty.
- **`AGENT_GRAPH is None`** — the api container's startup probe
  failed. `docker compose logs api --tail 50` usually shows the
  underlying cause (missing env var, Neo4j unreachable, etc.).

## License

Code is MIT-licensed.

**DrugBank data is NOT.** Any DrugBank-derived artifact — the XML
dump, the Neo4j database, `chroma_db/`, `gazetteer.pkl` — is covered
by the DrugBank academic license and cannot be redistributed. Only
code, tests, and synthesized demo-response strings are tracked in
this repository.
