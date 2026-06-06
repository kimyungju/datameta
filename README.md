# DataMeta

DataMeta is a localhost MVP for Shoppy, an e-commerce analytics and operations company.

It demonstrates:

- Git-versioned markdown knowledge for team-specific definitions and runbooks.
- RBAC over team folders and warehouse tables.
- ARR calculation disambiguation between Finance and Renewals definitions.
- Cross-team outlier flagging and owner resolution.
- Reusable analytics runbooks that produce chart and table outputs.
- A FastAPI REST API plus a Codex-compatible MCP JSON-RPC endpoint.

## Run

Backend:

```bash
cd backend
python3 -m uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:3000`.

Neo4j runs on Neo4j Aura (cloud). The MVP includes `backend/neo4j/schema.cypher` for the vector, full-text, and graph indexes; load it once against the Aura database before the first sync. The running localhost demo uses the same document/entity boundaries with a deterministic local retriever when Neo4j is not connected.

Reset seeded demo state:

```bash
python3 backend/scripts/reset_demo.py
```

OpenAI authoring configuration:

```bash
export OPENAI_API_KEY="..."
export DATAMETA_REASONING_MODEL="..."
# Optional alias: OPENAI_MODEL="..."
# Optional embedding metadata: DATAMETA_EMBEDDING_MODEL="..."
```

DataMeta uses the OpenAI Responses API to convert authoring comments into markdown front matter, body markdown, search terms, and Neo4j graph metadata. If `OPENAI_API_KEY` is set without `DATAMETA_REASONING_MODEL` or `OPENAI_MODEL`, authoring is blocked with a configuration error instead of silently using local heuristics. If no OpenAI key is set, the demo reports `local_deterministic` mode and uses the local fallback.

Optional Neo4j sync for committed knowledge:

Sync uses the official Neo4j Bolt driver (`neo4j` Python package), so the URL is a
Bolt URI pointing at Neo4j Aura.

```bash
# Neo4j Aura (note: Aura names the user and default database after the instance id):
export DATAMETA_NEO4J_URL="neo4j+s://<instance-id>.databases.neo4j.io"
export DATAMETA_NEO4J_USER="<instance-id>"          # often the instance id, not "neo4j"
export DATAMETA_NEO4J_PASSWORD="..."
export DATAMETA_NEO4J_DATABASE="<instance-id>"      # defaults to "neo4j" if unset
```

On localhost the backend reads these from `backend/.env` (gitignored). Load the index
definitions once with `backend/neo4j/schema.cypher` before the first sync.

When these are set, committed markdown documents are upserted into Neo4j with document, team, entity, search-term, and model-produced relationship metadata. The local markdown repository remains the source of truth for validation and fallback search.

Codex MCP configuration target:

```toml
[mcp_servers.datameta]
url = "http://127.0.0.1:8000/mcp"
bearer_token_env_var = "DATAMETA_MCP_TOKEN"
tool_timeout_sec = 120
```

If `DATAMETA_MCP_TOKEN` is unset, the local demo MCP endpoint allows unauthenticated calls.

Once configured, users can ask Codex plain-language questions such as:

- "What is the definition of ARR?"
- "Data point seems suspicious"

Codex should call `datameta_ask` first for conversational requests. It answers grounded definition questions with citations, and it either flags detailed data-quality concerns or asks for the missing table, subject, and description.
