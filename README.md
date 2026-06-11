# DataMeta

DataMeta is a localhost MVP of a knowledge layer over an enterprise's markdown repositories. The seeded demo corpus covers six incident-knowledge repos (security, legal, customer success, platform operations, vendor risk, data governance) plus three metric-definition repos (finance, renewals, sales).

It demonstrates:

- Git-versioned markdown knowledge for team-specific definitions and runbooks.
- RBAC over team repositories and warehouse tables.
- Multi-repo retrieval with citations (repository → folder → file funnel, hybrid local/Neo4j ranking).
- ARR calculation disambiguation: when the Finance, Renewals, and Sales definitions conflict, the app pauses and asks which one to use instead of guessing.
- Cross-team outlier flagging and owner resolution.
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

## MCP security and permission model

The `/mcp` endpoint exposes 16 tools. Their permission boundaries are explicit
and tested (`backend/tests/test_mcp_contract.py`):

- **Authentication** — when `DATAMETA_MCP_TOKEN` is set, every JSON-RPC call
  must carry `Authorization: Bearer <token>`; wrong or missing tokens get 401.
  Unset means unauthenticated localhost demo mode, by design.
- **RBAC scoping** — every retrieval/calculation tool takes a `user_id` and
  filters repositories, folders, files, and warehouse tables to that user's
  `read_teams` / `write_teams` / `tables`. A user asking for a file outside
  their teams gets a tool error, not data. Path traversal in
  `datameta_read_markdown_file` is rejected before any lookup.
- **Tool annotations** — every tool declares MCP annotations
  (`readOnlyHint`, `destructiveHint`, `openWorldHint`) so agent runtimes can
  gate approval UX. `datameta_commit_proposal` is the only tool marked
  destructive.
- **Two-phase writes** — knowledge changes go draft → validate → commit.
  `datameta_author_proposal` never writes; `datameta_commit_proposal` requires
  a validated proposal id and an explicit `confirm_overwrite` flag when it
  would replace an existing page.
- **Pause instead of guess** — when multiple teams define a metric
  differently, `datameta_ask` returns `requires_choice=true` with the visible
  definitions instead of picking one (human-in-the-loop disambiguation).
- **Audit trail** — every commit lands in the Git-backed corpus with author
  and timestamp; `datameta_history` exposes commits, flags, and proposal
  history for review.

The contract tests also pin registry consistency: every tool listed by
`tools/list` is dispatchable, every dispatchable tool is listed, and the
bootstrap payload advertises the same 16 tools as the server.
