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

Neo4j, once Docker Desktop is running:

```bash
docker compose up -d neo4j
```

The MVP includes `backend/neo4j/schema.cypher` for the planned vector, full-text, and graph indexes. The running localhost demo uses the same document/entity boundaries with a deterministic local retriever when Neo4j is not connected.

Reset seeded demo state:

```bash
python3 backend/scripts/reset_demo.py
```

Codex MCP configuration target:

```toml
[mcp_servers.datameta]
url = "http://127.0.0.1:8000/mcp"
bearer_token_env_var = "DATAMETA_MCP_TOKEN"
tool_timeout_sec = 120
```

If `DATAMETA_MCP_TOKEN` is unset, the local demo MCP endpoint allows unauthenticated calls.
