# Neo4j Hybrid RAG Implementation Plan (metadata-only embeddings)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Neo4j the primary embedding store for hybrid (vector + fulltext) retrieval and RAG, where **only Repository / Folder / Document metadata is embedded**, with a deterministic, literal-free local fallback for tests/dev.

**Architecture:** Embeddings are computed from each node's `metadata_text` (the structured `.datameta.md` fields) and stored on Repository, Folder, and Document nodes. **Chunks are NOT embedded** — they are synced with `path`/`heading`/`text` and retrieved via the chunk fulltext index. When Neo4j is configured, `multirepo_query` retrieves candidates via `db.index.vector.queryNodes` over the **Document** vector index plus `db.index.fulltext.queryNodes` over the **Chunk** fulltext index, merges them at the document level into a generic hybrid score, RBAC-filters by readable teams, and synthesizes the answer from that evidence (Document evidence carrying the best matching chunk's heading + snippet). When Neo4j is not configured, a deterministic local hybrid path produces the same evidence *shape* from the in-memory index. All Customer-A/Vendor-X/SLA hardcoded boosts are replaced with corpus-derived entity scoring.

**Tech Stack:** Python 3 (stdlib `urllib`), FastAPI, SQLite, Neo4j HTTP transactional endpoint (`/db/neo4j/tx/commit`), `unittest`/`pytest`.

**Scope decision — embeddings are metadata-only (overrides the original spec):**
- Per direction, **only repo/folder/file (Document) metadata is embedded**. This intentionally overrides the original spec's `Chunk.embedding`, the chunk vector index (item 3), and item 4's "especially Chunk indexes". Chunk retrieval is keyword/fulltext only.
- The current code already embeds + syncs repo/folder/document `metadata_text` and already defines repo/folder/document vector indexes and all fulltext indexes (including chunk fulltext). So no new embedding computation or new vector index is required — this plan adds a dimension guard, generic scoring/synthesis, the Neo4j read helper, and the Neo4j/local hybrid retrieval path.

**Key decisions (resolved before planning):**
- **Embedding dimensions:** OpenAI `text-embedding-3-large` = 3072 dims (production); local hash fallback = 256 dims (`LOCAL_EMBEDDING_DIMENSIONS`, tests/dev). Vector indexes are fixed at the configured dimension (default 3072, overridable by `DATAMETA_NEO4J_VECTOR_DIMENSIONS`). **The local 256-dim path never writes to or queries the vector index** unless the configured dimension is overridden to match.
- **Routing test:** The exact 4-repo shortlist for the Customer-A/Vendor-X query is **only achievable with real (OpenAI) embeddings** and is documented as the production target. Under the local hash embedding, generic corpus-derived scoring legitimately surfaces `data-governance` (its `customer-a-sla-evidence-checklist.md` is a real match). The local routing test therefore asserts a **weaker but real invariant**: the shortlist is a subset of readable repos and contains `vendor-risk-management` + `platform-operations`. (Confirmed empirically: the exact set is unrecoverable without hardcoded couplings, which the spec bans.)
- **RBAC in Neo4j path:** Filter in Cypher via a readable-team-name parameter, plus a Python post-filter safety net.

**Deterministic local-scorer outputs (measured, used for assertions below):**
- Counts: repositories=6, folders=18, files=54, chunks=61.
- Query `"Customer A availability SLA Vendor X"` → agreement files order: `customer-a`, `customer-b`, `customer-c` (customer-a first).
- Query `"Vendor X incident evidence for Customer A SLA review"` → first vendor file starts with `vendor-risk-management/vendor-x/`; `vendor-x-risk-register.md` cited; `vendor-y-export-delay.md` NOT cited.
- Query `"What is the Singapore pantry catering policy?"` → not answerable, no citations.
- Canonical query (`"Customer A says we missed their availability SLA during the Vendor X incident. What should we do?"`) under generic scoring → shortlisted repos = `{data-governance, platform-operations, security-incident-response, vendor-risk-management}`; answerable; citations include `platform-operations/incidents/vendor-x-2026-05-availability-incident.md`, `platform-operations/slo-measurement/customer-a-may-2026-availability.md`, `vendor-risk-management/vendor-x/vendor-x-risk-register.md`.

**Conventions:** All paths are repo-relative to `C:\NUS\Projects\datameta-natlv`. Run tests from `backend/` with `python -m pytest tests/test_datameta.py -v` (or `python -m unittest tests.test_datameta -v`).

---

## File Structure

- `backend/app/datameta.py` — all service logic (indexing, embeddings, sync, retrieval, synthesis). Single large existing file; follow its established style (private `_helpers`, stdlib `urllib`, dict-shaped results).
- `backend/neo4j/schema.cypher` — Neo4j constraints + fulltext + vector indexes.
- `backend/tests/test_datameta.py` — unittest suite; extend in place.

---

## Task 1: Dimension guard on metadata embedding sync

Extract the Neo4j sync statement-building into a testable `_build_neo4j_sync_statements`, and write `r/f/d.embedding` only when the produced embedding dimension matches the configured Neo4j vector-index dimension (so the 256-dim local fallback never corrupts a 3072-dim index). Chunk statements are unchanged (no embedding).

**Files:**
- Modify: `backend/app/datameta.py` — add `_neo4j_vector_dimensions`; refactor `_sync_multirepo_index_to_neo4j` (lines ~2166-2326)
- Test: `backend/tests/test_datameta.py`

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_datameta.py` inside `DataMetaServiceTest`:

```python
    def test_neo4j_vector_dimensions_defaults_and_override(self) -> None:
        self.assertEqual(3072, self.service._neo4j_vector_dimensions())
        with patch.dict("os.environ", {"DATAMETA_NEO4J_VECTOR_DIMENSIONS": "256"}):
            self.assertEqual(256, self.service._neo4j_vector_dimensions())

    def test_document_sync_writes_embedding_when_dimensions_match(self) -> None:
        self.service.index_repos()
        index = self.service._multirepo_index
        with patch.dict("os.environ", {"DATAMETA_NEO4J_VECTOR_DIMENSIONS": "256"}):
            statements = self.service._build_neo4j_sync_statements(index)
        document_statements = [s for s in statements if "MERGE (d:Document" in s["statement"]]
        self.assertTrue(document_statements)
        self.assertIn("d.embedding = $embedding", document_statements[0]["statement"])
        self.assertIn("embedding", document_statements[0]["parameters"])

    def test_document_sync_omits_embedding_when_dimensions_mismatch(self) -> None:
        self.service.index_repos()
        index = self.service._multirepo_index
        with patch.dict("os.environ", {"DATAMETA_NEO4J_VECTOR_DIMENSIONS": "3072"}):
            statements = self.service._build_neo4j_sync_statements(index)
        document_statements = [s for s in statements if "MERGE (d:Document" in s["statement"]]
        self.assertTrue(document_statements)
        self.assertNotIn("d.embedding", document_statements[0]["statement"])

    def test_chunk_sync_never_writes_embedding(self) -> None:
        self.service.index_repos()
        index = self.service._multirepo_index
        with patch.dict("os.environ", {"DATAMETA_NEO4J_VECTOR_DIMENSIONS": "256"}):
            statements = self.service._build_neo4j_sync_statements(index)
        chunk_statements = [s for s in statements if "MERGE (c:Chunk" in s["statement"]]
        self.assertTrue(chunk_statements)
        self.assertNotIn("c.embedding", chunk_statements[0]["statement"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_datameta.py -v -k "vector_dimensions or document_sync or chunk_sync_never"`
Expected: FAIL with `AttributeError: 'DataMetaService' object has no attribute '_neo4j_vector_dimensions'` / `_build_neo4j_sync_statements`.

- [ ] **Step 3: Add the dimension helper**

In `backend/app/datameta.py`, add this method right after `_neo4j_status` (after line ~2082):

```python
    def _neo4j_vector_dimensions(self) -> int:
        raw = configured_env_value("DATAMETA_NEO4J_VECTOR_DIMENSIONS")
        if raw and raw.isdigit():
            return int(raw)
        return 3072
```

- [ ] **Step 4: Extract `_build_neo4j_sync_statements` and apply the guard to repo/folder/document only**

In `_sync_multirepo_index_to_neo4j`, replace everything from `statements = []` (line ~2173) down to and including the chunk loop that ends at line ~2309 with a single call:

```python
        statements = self._build_neo4j_sync_statements(index)
```

So `_sync_multirepo_index_to_neo4j` keeps only: the `neo4j` config check, reading `url`/`user`/`password`, `statements = self._build_neo4j_sync_statements(index)`, and the existing batched POST loop (lines ~2310-2326) unchanged.

Add the new method immediately above `_sync_multirepo_index_to_neo4j`:

```python
    def _build_neo4j_sync_statements(self, index: dict[str, Any]) -> list[dict[str, Any]]:
        target_dims = self._neo4j_vector_dimensions()

        def embedding_clause(node_alias: str, embedding: list[float]) -> tuple[str, bool]:
            write = isinstance(embedding, list) and len(embedding) == target_dims
            return (f", {node_alias}.embedding = $embedding" if write else ""), write

        statements: list[dict[str, Any]] = []
        for item in index["repositories"].values():
            clause, write = embedding_clause("r", item.get("embedding", []))
            parameters = {
                "id": item["id"],
                "repository": item["repository"],
                "path": item["path"],
                "title": item["title"],
                "summary": item["summary"],
                "metadata_text": item["metadata_text"],
            }
            if write:
                parameters["embedding"] = item["embedding"]
            statements.append(
                {
                    "statement": (
                        "MERGE (r:Repository {id: $id}) "
                        "SET r.name = $repository, r.path = $path, r.title = $title, r.summary = $summary, "
                        "r.metadata_text = $metadata_text" + clause
                    ),
                    "parameters": parameters,
                }
            )
        for item in index["folders"].values():
            clause, write = embedding_clause("f", item.get("embedding", []))
            parameters = {
                "id": item["id"],
                "repository": item["repository"],
                "folder": item["folder"],
                "path": item["path"],
                "title": item["title"],
                "summary": item["summary"],
                "metadata_text": item["metadata_text"],
            }
            if write:
                parameters["embedding"] = item["embedding"]
            statements.append(
                {
                    "statement": (
                        "MATCH (r:Repository {name: $repository}) "
                        "MERGE (f:Folder {id: $id}) "
                        "SET f.repository = $repository, f.folder = $folder, f.path = $path, f.title = $title, "
                        "f.summary = $summary, f.metadata_text = $metadata_text" + clause + " "
                        "MERGE (r)-[:HAS_FOLDER]->(f)"
                    ),
                    "parameters": parameters,
                }
            )
        for item in index["files"].values():
            doc = item["doc"]
            clause, write = embedding_clause("d", item.get("embedding", []))
            parameters = {
                "folder_id": f"{doc['repository']}-{doc['folder']}-folder-metadata",
                "id": item["id"],
                "repository": item["repository"],
                "folder": item["folder"],
                "path": item["path"],
                "type": doc["type"],
                "title": item["title"],
                "summary": item["summary"],
                "metadata_text": item["metadata_text"],
                "commit_hash": (doc.get("commit") or {}).get("hash"),
            }
            if write:
                parameters["embedding"] = item["embedding"]
            statements.append(
                {
                    "statement": (
                        "MATCH (f:Folder {id: $folder_id}) "
                        "MERGE (d:Document {id: $id}) "
                        "SET d.repository = $repository, d.folder = $folder, d.path = $path, d.type = $type, "
                        "d.title = $title, d.summary = $summary, d.metadata_text = $metadata_text, "
                        "d.commit_hash = $commit_hash" + clause + " "
                        "MERGE (f)-[:HAS_DOCUMENT]->(d)"
                    ),
                    "parameters": parameters,
                }
            )
            statements.append(
                {
                    "statement": (
                        "MATCH (d:Document {id: $id}) "
                        "MERGE (t:Team {name: $team}) "
                        "MERGE (d)-[:OWNED_BY]->(t)"
                    ),
                    "parameters": {"id": item["id"], "team": doc["team"]},
                }
            )
            for customer in doc.get("customers", []):
                statements.append(
                    {
                        "statement": (
                            "MATCH (d:Document {id: $id}) "
                            "MERGE (c:Customer {name: $name}) "
                            "MERGE (d)-[:APPLIES_TO_CUSTOMER]->(c)"
                        ),
                        "parameters": {"id": item["id"], "name": customer},
                    }
                )
            for vendor in doc.get("vendors", []):
                statements.append(
                    {
                        "statement": (
                            "MATCH (d:Document {id: $id}) "
                            "MERGE (v:Vendor {name: $name}) "
                            "MERGE (d)-[:REFERENCES_VENDOR]->(v)"
                        ),
                        "parameters": {"id": item["id"], "name": vendor},
                    }
                )
            for sla in doc.get("slas", []):
                statements.append(
                    {
                        "statement": (
                            "MATCH (d:Document {id: $id}) "
                            "MERGE (s:SLA {id: $name}) "
                            "SET s.name = $name "
                            "MERGE (d)-[:REFERENCES_SLA]->(s)"
                        ),
                        "parameters": {"id": item["id"], "name": sla},
                    }
                )
            for incident in doc.get("incidents", []):
                statements.append(
                    {
                        "statement": (
                            "MATCH (d:Document {id: $id}) "
                            "MERGE (i:Incident {id: $name}) "
                            "SET i.name = $name "
                            "MERGE (d)-[:REFERENCES_INCIDENT]->(i)"
                        ),
                        "parameters": {"id": item["id"], "name": incident},
                    }
                )
        for chunk in index["chunks"]:
            statements.append(
                {
                    "statement": (
                        "MATCH (d:Document {id: $document_id}) "
                        "MERGE (c:Chunk {id: $id}) "
                        "SET c.path = $path, c.heading = $heading, c.text = $text "
                        "MERGE (d)-[:HAS_CHUNK]->(c)"
                    ),
                    "parameters": {
                        "id": chunk["id"],
                        "document_id": chunk["document_id"],
                        "path": chunk["path"],
                        "heading": chunk["heading"],
                        "text": chunk["text"],
                    },
                }
            )
        return statements
```

> Chunks are synced with `path`/`heading`/`text` only — no embedding (metadata-only embedding scope).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_datameta.py -v -k "vector_dimensions or document_sync or chunk_sync_never"`
Expected: PASS (4 tests).

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/test_datameta.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/datameta.py backend/tests/test_datameta.py
git commit -m "feat: guard metadata embedding sync by vector dimension"
```

---

## Task 2: Document the fixed-dimension constraint; lock metadata-only index shape

The current schema already defines Repository/Folder/Document vector indexes and Repository/Folder/Document/Chunk fulltext indexes — exactly the metadata-only shape we want. This task only documents the fixed-dimension constraint and locks (via test) that there is **no** chunk vector index.

**Files:**
- Modify: `backend/neo4j/schema.cypher`
- Test: `backend/tests/test_datameta.py`

- [ ] **Step 1: Write the failing test**

```python
    def test_schema_is_metadata_only_vector_with_chunk_fulltext(self) -> None:
        from pathlib import Path as _Path

        schema = (_Path(__file__).resolve().parents[1] / "neo4j" / "schema.cypher").read_text(encoding="utf-8")
        # vector indexes on metadata nodes only
        self.assertIn("datameta_repository_embedding", schema)
        self.assertIn("datameta_folder_embedding", schema)
        self.assertIn("datameta_document_embedding", schema)
        # NO chunk vector index (metadata-only embedding scope)
        self.assertNotIn("datameta_chunk_embedding", schema)
        # fulltext indexes (incl. chunk) are preserved for keyword search
        self.assertIn("datameta_document_fulltext", schema)
        self.assertIn("datameta_chunk_fulltext", schema)
        # fixed-dimension constraint is documented
        self.assertIn("DATAMETA_NEO4J_VECTOR_DIMENSIONS", schema)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_datameta.py::DataMetaServiceTest::test_schema_is_metadata_only_vector_with_chunk_fulltext -v`
Expected: FAIL on `assertIn("DATAMETA_NEO4J_VECTOR_DIMENSIONS", schema)` (the constraint comment is not present yet).

- [ ] **Step 3: Add the dimension-constraint documentation to `schema.cypher`**

Append to the end of `backend/neo4j/schema.cypher`:

```cypher

// Vector dimension constraint:
// Embeddings are metadata-only: Repository, Folder, and Document nodes carry an `embedding`
// computed from their metadata_text. Chunks are NOT embedded (no datameta_chunk_embedding
// index); chunk retrieval uses datameta_chunk_fulltext for keyword search.
// All datameta_*_embedding vector indexes are fixed at 3072 dimensions to match the production
// embedding model (OpenAI text-embedding-3-large). If you change the embedding model, recreate
// these indexes with the matching dimension and set DATAMETA_NEO4J_VECTOR_DIMENSIONS so sync
// writes only matching-dimension vectors. The 256-dimension local hash fallback (tests/dev) is
// NOT written to these indexes unless DATAMETA_NEO4J_VECTOR_DIMENSIONS is overridden to 256.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_datameta.py::DataMetaServiceTest::test_schema_is_metadata_only_vector_with_chunk_fulltext -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/neo4j/schema.cypher backend/tests/test_datameta.py
git commit -m "docs: document metadata-only vector index dimension constraint"
```

---

## Task 3: Generic RAG answer synthesis (remove canned Customer-A/Vendor-X branch)

Replace the hardcoded Customer-A/Vendor-X answer with synthesis from retrieved evidence: OpenAI synthesis when configured, otherwise an extractive summary; "not answerable" when evidence is weak. This task keeps the existing phrase boosts in place (scoring is replaced in Task 4), so the suite stays green between commits.

**Files:**
- Modify: `backend/app/datameta.py` — `_synthesize_multirepo_answer` (lines ~2607-2631); add `_synthesize_answer_with_openai`
- Modify: `backend/tests/test_datameta.py` — `test_customer_a_vendor_x_sla_query_routes_to_expected_repositories`

- [ ] **Step 1: Update the existing test to stop asserting canned percentages**

In `backend/tests/test_datameta.py`, replace the body of `test_customer_a_vendor_x_sla_query_routes_to_expected_repositories` (currently lines ~91-104) with:

```python
    def test_customer_a_vendor_x_sla_query_routes_to_expected_repositories(self) -> None:
        result = self.service.multirepo_query(
            "junior.analyst",
            "Customer A says we missed their availability SLA during the Vendor X incident. What should we do?",
            True,
        )
        repos = {repo["repository"] for repo in result["shortlisted_repositories"]}
        self.assertEqual(
            {"vendor-risk-management", "customer-success-ops", "legal-contracts", "platform-operations"},
            repos,
        )
        self.assertTrue(result["answerable"])
        self.assertTrue(result["citations"])
```

> The exact 4-repo `assertEqual` still passes here because Task 3 leaves the phrase boosts intact; it is weakened in Task 4 when scoring becomes generic. The `99.72`/`99.90` string assertions are removed because they came from the canned branch deleted in this task.

- [ ] **Step 2: Replace `_synthesize_multirepo_answer` and add the OpenAI synthesizer**

In `backend/app/datameta.py`, replace the entire `_synthesize_multirepo_answer` method (lines ~2607-2631) with a generic version that accepts evidence/finding items carrying `title`/`summary`/`snippet`:

```python
    def _synthesize_multirepo_answer(self, question: str, items: list[dict[str, Any]]) -> tuple[bool, str]:
        if not items:
            return False, (
                "Not answerable from available knowledge. DataMeta did not find a relevant repository, folder, "
                "or file path in the configured corpus, so it will not infer an answer."
            )
        text = " ".join(
            f"{item.get('title', '')} {item.get('summary', '')} {item.get('snippet', '')}" for item in items
        ).lower()
        direct_terms = sum(1 for term in tokenize(question) if term in tokenize(text))
        if direct_terms < 3:
            return False, (
                "Not answerable from available knowledge. DataMeta found some weak overlap, but the selected "
                "evidence did not directly support an answer."
            )
        config = model_config()
        if config["mode"] == "openai_configured" and configured_env_value("DATAMETA_DISABLE_OPENAI_SUBAGENTS") != "1":
            try:
                return True, self._synthesize_answer_with_openai(question, items, config)
            except Exception:
                pass
        summaries = "; ".join(
            f"{item.get('title', '')}: {item.get('summary') or item.get('snippet', '')}".strip()
            for item in items[:5]
        )
        return True, f"Based on the retrieved evidence: {summaries}."

    def _synthesize_answer_with_openai(self, question: str, items: list[dict[str, Any]], config: dict[str, Any]) -> str:
        client = self._authoring_client(config)
        evidence = [
            {
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "snippet": item.get("snippet", ""),
                "path": (item.get("citation") or {}).get("file_path") or item.get("path", ""),
            }
            for item in items[:8]
        ]
        system = (
            "You are DataMeta's retrieval answer synthesizer. Use ONLY the supplied evidence to answer. "
            "Do not use outside knowledge. If the evidence does not support an answer, say it is not answerable."
        )
        user = json.dumps({"question": question, "evidence": evidence}, indent=2, sort_keys=True)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        }
        result = client.structured_json(system=system, user=user, schema_name="datameta_rag_answer", schema=schema)
        answer = str(result.get("answer", "")).strip()
        if not answer:
            raise RuntimeError("OpenAI synthesis returned an empty answer")
        return answer
```

- [ ] **Step 3: Add a test proving canned text is gone and synthesis is evidence-based**

```python
    def test_answer_is_extractive_and_has_no_canned_customer_a_paragraph(self) -> None:
        result = self.service.multirepo_query(
            "junior.analyst",
            "Customer A says we missed their availability SLA during the Vendor X incident. What should we do?",
            True,
        )
        self.assertTrue(result["answerable"])
        # the deleted canned branch began with this exact phrase
        self.assertNotIn("Customer A's complaint is answerable from the available knowledge", result["answer"])
        self.assertTrue(result["answer"].startswith("Based on the retrieved evidence:"))
```

- [ ] **Step 4: Run the new and updated tests**

Run: `python -m pytest tests/test_datameta.py -v -k "routes_to_expected_repositories or extractive_and_has_no_canned"`
Expected: PASS (both).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/test_datameta.py -v`
Expected: all PASS. (`test_final_answer_cites_actual_markdown_files` still passes because phrase boosts remain until Task 4.)

- [ ] **Step 6: Commit**

```bash
git add backend/app/datameta.py backend/tests/test_datameta.py
git commit -m "feat: synthesize RAG answers from retrieved evidence, remove canned branch"
```

---

## Task 4: Replace hardcoded boosts with corpus-derived entity scoring

Replace `_phrase_boost` and `_doc_directly_supports_query` with generic versions driven by `index["entities"]` (customers, vendors, slas, incidents) — no literal entity names. Update the two tests whose expectations depended on the hardcoded couplings.

**Files:**
- Modify: `backend/app/datameta.py` — `_phrase_boost` (lines ~2375-2414) and `_doc_directly_supports_query` (lines ~2466-2489)
- Modify: `backend/tests/test_datameta.py` — `test_customer_a_vendor_x_sla_query_routes_to_expected_repositories`, `test_final_answer_cites_actual_markdown_files`

- [ ] **Step 1: Write the failing guard test (no hardcoded literals)**

```python
    def test_scoring_methods_contain_no_hardcoded_entity_literals(self) -> None:
        import inspect

        source = (
            inspect.getsource(DataMetaService._phrase_boost)
            + inspect.getsource(DataMetaService._doc_directly_supports_query)
        ).lower()
        for literal in ("customer a", "customer b", "customer c", "vendor x", "vendor y", "service credit"):
            self.assertNotIn(literal, source)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_datameta.py::DataMetaServiceTest::test_scoring_methods_contain_no_hardcoded_entity_literals -v`
Expected: FAIL (current `_phrase_boost` contains `"customer a"`, `"vendor x"`, etc.).

- [ ] **Step 3: Replace `_phrase_boost` with corpus-derived entity boost**

Replace the entire `_phrase_boost` method (lines ~2375-2414) with:

```python
    def _phrase_boost(self, query: str, text: str) -> float:
        index = self._multirepo_index
        if not index:
            return 0.0
        entities = index.get("entities", {})
        query_phrase = query.lower().replace("-", " ")
        text_phrase = text.lower().replace("-", " ")
        boost = 0.0
        for family in ("customers", "vendors", "slas", "incidents"):
            members = [member.lower().replace("-", " ") for member in entities.get(family, [])]
            requested = [member for member in members if member and member in query_phrase]
            if not requested:
                continue
            for member in members:
                if not member:
                    continue
                if member in requested:
                    if member in text_phrase:
                        boost += 2.5
                elif member in text_phrase:
                    boost -= 1.0
        return boost
```

- [ ] **Step 4: Replace `_doc_directly_supports_query` with corpus-derived support filter**

Replace the entire `_doc_directly_supports_query` method (lines ~2466-2489) with:

```python
    def _doc_directly_supports_query(self, query: str, doc: dict[str, Any]) -> bool:
        index = self._multirepo_index
        entities = index.get("entities", {}) if index else {}
        query_phrase = query.lower().replace("-", " ")
        text_phrase = f"{self._metadata_text(doc)} {doc['body']}".lower().replace("-", " ")
        for family, doc_key in (("customers", "customers"), ("vendors", "vendors")):
            members = [member.lower().replace("-", " ") for member in entities.get(family, [])]
            requested = [member for member in members if member and member in query_phrase]
            if not requested:
                continue
            doc_members = [value.lower().replace("-", " ") for value in doc.get(doc_key, [])]
            if doc_members and not any(member in doc_members for member in requested):
                return False
            if not any(member in text_phrase for member in requested):
                return False
        return True
```

- [ ] **Step 5: Weaken the routing assertion in `test_customer_a_vendor_x_sla_query_routes_to_expected_repositories`**

Replace the `repos = {...}` / `self.assertEqual(...)` block (the version written in Task 3) with the documented weaker-but-real local invariant:

```python
    def test_customer_a_vendor_x_sla_query_routes_to_expected_repositories(self) -> None:
        result = self.service.multirepo_query(
            "junior.analyst",
            "Customer A says we missed their availability SLA during the Vendor X incident. What should we do?",
            True,
        )
        repos = {repo["repository"] for repo in result["shortlisted_repositories"]}
        readable = {
            "security-incident-response",
            "legal-contracts",
            "customer-success-ops",
            "platform-operations",
            "vendor-risk-management",
            "data-governance",
        }
        # Local hash-embedding fallback: exact 4-repo routing is a production (OpenAI) target.
        # Here we assert a real invariant: shortlist is readable and includes the two repos that
        # survive every generic configuration.
        self.assertTrue(repos.issubset(readable))
        self.assertIn("vendor-risk-management", repos)
        self.assertIn("platform-operations", repos)
        self.assertTrue(result["answerable"])
        self.assertTrue(result["citations"])
```

- [ ] **Step 6: Rewrite `test_final_answer_cites_actual_markdown_files` to the deterministic generic citations**

Replace its body (currently lines ~167-178) with:

```python
    def test_final_answer_cites_actual_markdown_files(self) -> None:
        result = self.service.multirepo_query(
            "junior.analyst",
            "Customer A says we missed their availability SLA during the Vendor X incident. What should we do?",
            True,
        )
        cited_paths = {citation["file_path"] for citation in result["citations"]}
        # Deterministic under generic corpus-derived scoring with the local hash embedding.
        self.assertIn("platform-operations/incidents/vendor-x-2026-05-availability-incident.md", cited_paths)
        self.assertIn("platform-operations/slo-measurement/customer-a-may-2026-availability.md", cited_paths)
        self.assertIn("vendor-risk-management/vendor-x/vendor-x-risk-register.md", cited_paths)
        # every citation points at a real markdown file in the corpus
        all_paths = {doc["path"] for doc in self.service.all_documents()}
        self.assertTrue(cited_paths.issubset(all_paths))
```

- [ ] **Step 7: Run the affected tests**

Run: `python -m pytest tests/test_datameta.py -v -k "no_hardcoded_entity_literals or routes_to_expected_repositories or final_answer_cites or ranks_above_customer_b or vendor_x_files_rank or unanswerable_query"`
Expected: PASS. Specifically:
- `test_scoring_methods_contain_no_hardcoded_entity_literals` PASS
- `test_customer_a_sla_ranks_above_customer_b_and_c_distractors` PASS (customer-a first — verified)
- `test_vendor_x_files_rank_and_cite_above_vendor_y_distractors` PASS (vendor-x first; vendor-y excluded — verified)
- `test_unanswerable_query_does_not_invent_facts` PASS (verified)

- [ ] **Step 8: Run the full suite and confirm Q1 answerable**

Run: `python -m pytest tests/test_datameta.py -v`
Expected: all PASS. (Verification note: Q1 `answerable` now rides on `direct_terms >= 3` rather than the deleted canned branch — confirm via this run rather than assuming.)

- [ ] **Step 9: Commit**

```bash
git add backend/app/datameta.py backend/tests/test_datameta.py
git commit -m "refactor: corpus-derived entity scoring, drop hardcoded boosts"
```

---

## Task 5: Neo4j read helper (`_neo4j_query`)

Add a transactional read helper that POSTs Cypher to `/db/neo4j/tx/commit` and returns rows as dicts. Reuses the auth pattern from sync.

**Files:**
- Modify: `backend/app/datameta.py` — add `_neo4j_query` (after `_neo4j_vector_dimensions`)
- Test: `backend/tests/test_datameta.py`

- [ ] **Step 1: Write the failing tests (mocked HTTP)**

```python
    def test_neo4j_query_parses_rows_when_configured(self) -> None:
        import io
        import json as _json

        payload = {
            "results": [
                {
                    "columns": ["id", "score"],
                    "data": [{"row": ["chunk-1", 0.9]}, {"row": ["chunk-2", 0.5]}],
                }
            ],
            "errors": [],
        }

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with patch.dict(
            "os.environ",
            {
                "DATAMETA_NEO4J_URL": "http://neo4j.test",
                "DATAMETA_NEO4J_USER": "neo4j",
                "DATAMETA_NEO4J_PASSWORD": "secret",
            },
        ):
            with patch("app.datameta.urllib.request.urlopen", return_value=_Resp(_json.dumps(payload).encode())):
                rows = self.service._neo4j_query("RETURN 1", {})
        self.assertEqual([{"id": "chunk-1", "score": 0.9}, {"id": "chunk-2", "score": 0.5}], rows)

    def test_neo4j_query_returns_none_when_not_configured(self) -> None:
        self.assertIsNone(self.service._neo4j_query("RETURN 1", {}))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_datameta.py -v -k "neo4j_query"`
Expected: FAIL with `AttributeError: ... '_neo4j_query'`.

- [ ] **Step 3: Implement `_neo4j_query`**

Add after `_neo4j_vector_dimensions` in `backend/app/datameta.py`:

```python
    def _neo4j_query(self, statement: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]] | None:
        if not self._neo4j_status()["configured"]:
            return None
        url = configured_env_value("DATAMETA_NEO4J_URL") or ""
        user = configured_env_value("DATAMETA_NEO4J_USER") or ""
        password = configured_env_value("DATAMETA_NEO4J_PASSWORD") or ""
        token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        endpoint = f"{url.rstrip('/')}/db/neo4j/tx/commit"
        body = {
            "statements": [
                {"statement": statement, "parameters": parameters or {}, "resultDataContents": ["row"]}
            ]
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError:
            return None
        if payload.get("errors"):
            return None
        results = payload.get("results") or []
        if not results:
            return []
        columns = results[0].get("columns", [])
        rows: list[dict[str, Any]] = []
        for entry in results[0].get("data", []):
            row = entry.get("row", [])
            rows.append({columns[i]: row[i] for i in range(min(len(columns), len(row)))})
        return rows
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_datameta.py -v -k "neo4j_query"`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/datameta.py backend/tests/test_datameta.py
git commit -m "feat: add Neo4j transactional read helper"
```

---

## Task 6: Neo4j hybrid retrieval (document vector + chunk fulltext) + local mirror, wired into `multirepo_query`

Add `_neo4j_hybrid_retrieve` (Document vector search + Chunk fulltext search, RBAC-filtered, merged at the document level into a generic hybrid score), `_merge_hybrid_rows`, and `_local_evidence_from_findings`. Wire a dispatcher into `multirepo_query` that prefers Neo4j when configured and records `retrieval_source`.

**Files:**
- Modify: `backend/app/datameta.py` — add 3 methods; modify `multirepo_query` (lines ~2636-2718)
- Test: `backend/tests/test_datameta.py`

- [ ] **Step 1: Write the failing tests**

```python
    def test_local_retrieval_source_is_local_hybrid(self) -> None:
        result = self.service.multirepo_query(
            "junior.analyst",
            "Vendor X incident evidence for Customer A SLA review",
            True,
        )
        self.assertEqual("local_hybrid", result["retrieval_source"])
        self.assertEqual("local_hybrid", result["trace"]["retrieval_source"])
        evidence = result["retrieval"]["evidence"]
        self.assertTrue(evidence)
        for item in evidence:
            for key in ("path", "heading", "snippet", "score", "vector_score", "keyword_score", "citation"):
                self.assertIn(key, item)

    def test_neo4j_hybrid_retrieval_merges_document_vector_and_chunk_fulltext(self) -> None:
        doc_path = "platform-operations/incidents/vendor-x-2026-05-availability-incident.md"
        vector_rows = [
            {
                "document_id": "inc-vendor-x-2026-05-20",
                "path": doc_path,
                "title": "Vendor X Availability Incident on 2026-05-20",
                "summary": "Vendor X token validation degradation.",
                "team": "platform-operations",
                "score": 0.92,
            }
        ]
        chunk_rows = [
            {
                "document_id": "inc-vendor-x-2026-05-20",
                "path": doc_path,
                "title": "Vendor X Availability Incident on 2026-05-20",
                "summary": "Vendor X token validation degradation.",
                "heading": "Timeline",
                "text": "Vendor X token validation degradation caused elevated public API errors for Customer A.",
                "team": "platform-operations",
                "score": 3.1,
            }
        ]

        def fake_query(statement, parameters=None):
            self.assertIn("admin", parameters)
            self.assertIn("teams", parameters)  # RBAC param is passed
            if "db.index.vector.queryNodes" in statement:
                return vector_rows
            if "db.index.fulltext.queryNodes" in statement:
                return chunk_rows
            return []

        with patch.dict(
            "os.environ",
            {
                "DATAMETA_NEO4J_URL": "http://neo4j.test",
                "DATAMETA_NEO4J_USER": "neo4j",
                "DATAMETA_NEO4J_PASSWORD": "secret",
                "DATAMETA_NEO4J_VECTOR_DIMENSIONS": "256",  # match local 256-dim query embedding
            },
        ):
            with patch.object(self.service, "_neo4j_query", side_effect=fake_query):
                result = self.service.multirepo_query(
                    "junior.analyst",
                    "Vendor X incident evidence for Customer A SLA review",
                    True,
                )
        self.assertEqual("neo4j_hybrid", result["retrieval_source"])
        self.assertEqual("neo4j_hybrid", result["trace"]["retrieval_source"])
        evidence = result["retrieval"]["evidence"]
        self.assertEqual(1, len(evidence))
        item = evidence[0]
        self.assertGreater(item["vector_score"], 0.0)
        self.assertGreater(item["keyword_score"], 0.0)
        self.assertGreater(item["score"], 0.0)
        self.assertEqual("Timeline", item["heading"])  # snippet/heading come from the chunk hit
        cited = {c["file_path"] for c in result["citations"]}
        self.assertIn(doc_path, cited)

    def test_neo4j_hybrid_respects_rbac_for_unreadable_teams(self) -> None:
        captured = {}

        def fake_query(statement, parameters=None):
            captured["teams"] = parameters.get("teams")
            captured["admin"] = parameters.get("admin")
            return []

        with patch.dict(
            "os.environ",
            {
                "DATAMETA_NEO4J_URL": "http://neo4j.test",
                "DATAMETA_NEO4J_USER": "neo4j",
                "DATAMETA_NEO4J_PASSWORD": "secret",
                "DATAMETA_NEO4J_VECTOR_DIMENSIONS": "256",
            },
        ):
            with patch.object(self.service, "_neo4j_query", side_effect=fake_query):
                self.service.multirepo_query("leah.legal", "Customer A SLA", True)
        self.assertFalse(captured["admin"])
        self.assertNotIn("security-incident-response", captured["teams"])
        self.assertIn("legal-contracts", captured["teams"])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_datameta.py -v -k "retrieval_source or neo4j_hybrid or rbac_for_unreadable"`
Expected: FAIL with `KeyError: 'retrieval_source'` (and `AttributeError` for the new methods once reached).

- [ ] **Step 3: Add the retrieval methods**

Add these three methods to `backend/app/datameta.py` just above `datameta_multirepo_query` (line ~2633):

```python
    def _merge_hybrid_rows(
        self, vector_rows: list[dict[str, Any]], chunk_rows: list[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        vector_high = max((float(r.get("score") or 0.0) for r in vector_rows), default=0.0)
        chunk_high = max((float(r.get("score") or 0.0) for r in chunk_rows), default=0.0)

        def blank(row: dict[str, Any]) -> dict[str, Any]:
            return {
                "document_id": row.get("document_id"),
                "path": row.get("path"),
                "title": row.get("title"),
                "summary": row.get("summary"),
                "heading": None,
                "text": None,
                "vector_score": 0.0,
                "keyword_score": 0.0,
            }

        docs: dict[str, dict[str, Any]] = {}
        for row in vector_rows:
            entry = docs.setdefault(row.get("document_id"), blank(row))
            entry["vector_score"] = (float(row.get("score") or 0.0) / vector_high) if vector_high else 0.0
        for row in chunk_rows:
            entry = docs.setdefault(row.get("document_id"), blank(row))
            normalized = (float(row.get("score") or 0.0) / chunk_high) if chunk_high else 0.0
            if normalized >= entry["keyword_score"]:
                # keep the best-scoring chunk as the snippet/heading source for this document
                entry["keyword_score"] = normalized
                entry["heading"] = row.get("heading")
                entry["text"] = row.get("text")
            entry["path"] = entry["path"] or row.get("path")
            entry["title"] = entry["title"] or row.get("title")
            entry["summary"] = entry["summary"] or row.get("summary")

        evidence: list[dict[str, Any]] = []
        for entry in docs.values():
            vector_score = entry["vector_score"]
            keyword_score = entry["keyword_score"]
            score = round(vector_score * 0.6 + keyword_score * 0.4, 4)
            body = entry["text"] or entry["summary"] or ""
            text = re.sub(r"\s+", " ", body).strip()
            snippet = (text[:317].rstrip() + "...") if len(text) > 320 else text
            heading = entry["heading"] or entry["title"] or ""
            path = entry["path"] or ""
            parts = path.split("/")
            evidence.append(
                {
                    "path": path,
                    "heading": heading,
                    "snippet": snippet,
                    "title": entry["title"] or path,
                    "summary": entry["summary"] or "",
                    "score": score,
                    "vector_score": round(vector_score, 4),
                    "keyword_score": round(keyword_score, 4),
                    "citation": {
                        "repository": parts[0] if parts else "",
                        "folder": "/".join(parts[1:-1]) if len(parts) > 2 else "",
                        "file_path": path,
                        "path": path,
                        "heading": heading,
                        "snippet": snippet,
                        "document_id": entry["document_id"],
                        "title": entry["title"] or path,
                    },
                }
            )
        evidence.sort(key=lambda item: item["score"], reverse=True)
        return evidence[:limit]

    def _neo4j_hybrid_retrieve(
        self, user: User, query: str, query_embedding: list[float], limit: int = 8
    ) -> dict[str, Any] | None:
        if not self._neo4j_status()["configured"]:
            return None
        if len(query_embedding) != self._neo4j_vector_dimensions():
            # Dimension mismatch (e.g. local 256-dim vs a 3072 index): cannot query the vector index.
            return None
        is_admin = user.is_admin
        teams = list(user.read_teams)
        vector_rows = self._neo4j_query(
            "CALL db.index.vector.queryNodes('datameta_document_embedding', $k, $vec) YIELD node, score "
            "MATCH (node)-[:OWNED_BY]->(t:Team) "
            "WHERE $admin OR t.name IN $teams "
            "RETURN node.id AS document_id, node.path AS path, node.title AS title, node.summary AS summary, "
            "t.name AS team, score AS score",
            {"k": limit * 2, "vec": query_embedding, "admin": is_admin, "teams": teams},
        ) or []
        chunk_rows = self._neo4j_query(
            "CALL db.index.fulltext.queryNodes('datameta_chunk_fulltext', $q) YIELD node, score "
            "MATCH (d:Document)-[:HAS_CHUNK]->(node) "
            "MATCH (d)-[:OWNED_BY]->(t:Team) "
            "WHERE $admin OR t.name IN $teams "
            "RETURN d.id AS document_id, d.path AS path, d.title AS title, d.summary AS summary, "
            "node.heading AS heading, node.text AS text, t.name AS team, score AS score",
            {"q": query, "admin": is_admin, "teams": teams},
        ) or []
        if not is_admin:
            allowed = set(teams)
            vector_rows = [r for r in vector_rows if r.get("team") in allowed]
            chunk_rows = [r for r in chunk_rows if r.get("team") in allowed]
        evidence = self._merge_hybrid_rows(vector_rows, chunk_rows, limit)
        return {"source": "neo4j_hybrid", "evidence": evidence}

    def _local_evidence_from_findings(
        self, findings: list[dict[str, Any]], shortlisted_files: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        hybrid_by_path = {item["path"]: (item.get("hybrid") or {}) for item in shortlisted_files}
        evidence: list[dict[str, Any]] = []
        for finding in findings:
            hybrid = hybrid_by_path.get(finding["file_path"], {})
            evidence.append(
                {
                    "path": finding["file_path"],
                    "heading": finding["heading"],
                    "snippet": finding["snippet"],
                    "title": finding["title"],
                    "summary": finding["summary"],
                    "score": hybrid.get("score", 0.0),
                    "vector_score": hybrid.get("vector_score", 0.0),
                    "keyword_score": hybrid.get("keyword_score", 0.0),
                    "citation": finding["citation"],
                }
            )
        return evidence
```

- [ ] **Step 4: Wire the dispatcher into `multirepo_query`**

In `multirepo_query`, locate the block that computes `findings`, `answerable`, `answer_text`, `citations`, and `shortlisted_files` (currently lines ~2683-2690):

```python
        findings = [finding for folder in folder_findings for finding in folder["findings"]]
        answerable, answer_text = self._synthesize_multirepo_answer(query, findings)
        citations = [finding["citation"] for finding in findings] if answerable else []
        shortlisted_files = [
            file_payload
            for folder in folder_findings
            for file_payload in folder["selected_files"]
        ]
```

Replace it with (compute `shortlisted_files` first; then dispatch retrieval):

```python
        findings = [finding for folder in folder_findings for finding in folder["findings"]]
        shortlisted_files = [
            file_payload
            for folder in folder_findings
            for file_payload in folder["selected_files"]
        ]
        neo4j_retrieval = self._neo4j_hybrid_retrieve(user, query, query_embedding)
        if neo4j_retrieval and neo4j_retrieval.get("evidence"):
            retrieval = neo4j_retrieval
            retrieval_source = "neo4j_hybrid"
            evidence = retrieval["evidence"]
            answerable, answer_text = self._synthesize_multirepo_answer(query, evidence)
            citations = [item["citation"] for item in evidence] if answerable else []
        else:
            retrieval_source = "local_hybrid"
            evidence = self._local_evidence_from_findings(findings, shortlisted_files)
            retrieval = {"source": "local_hybrid", "evidence": evidence}
            answerable, answer_text = self._synthesize_multirepo_answer(query, findings)
            citations = [finding["citation"] for finding in findings] if answerable else []
```

In the `trace = {...}` dict (currently lines ~2693-2706), add a `retrieval_source` entry next to `"neo4j": index["neo4j"],`:

```python
                "neo4j": index["neo4j"],
                "retrieval_source": retrieval_source,
```

In the returned result dict (currently lines ~2707-2718), add two keys next to `"answer": answer_text,`:

```python
            "answer": answer_text,
            "retrieval_source": retrieval_source,
            "retrieval": retrieval,
```

- [ ] **Step 5: Run the new tests**

Run: `python -m pytest tests/test_datameta.py -v -k "retrieval_source or neo4j_hybrid or rbac_for_unreadable"`
Expected: PASS (3 tests).

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/test_datameta.py -v`
Expected: all PASS (local-path tests still pass; `retrieval_source` defaults to `local_hybrid` when Neo4j is not configured).

- [ ] **Step 7: Commit**

```bash
git add backend/app/datameta.py backend/tests/test_datameta.py
git commit -m "feat: Neo4j hybrid (document vector + chunk fulltext) retrieval with RBAC and local mirror"
```

---

## Final verification

- [ ] **Run the complete suite one more time**

Run: `python -m pytest tests/test_datameta.py -v`
Expected: ALL PASS.

- [ ] **Acceptance criteria checklist (adjusted for metadata-only embedding scope):**
  - [ ] `datameta_index_repos(force=True, sync_neo4j=True)` writes embeddings to Neo4j for Repository, Folder, and Document (metadata-only; production OpenAI 3072-dim). Chunks are synced with path/heading/text, no embedding. — Task 1.
  - [ ] Neo4j has vector indexes for Repository, Folder, Document (no chunk vector index); fulltext indexes preserved including chunk; dimension constraint documented. — Task 2.
  - [ ] `datameta_multirepo_query` uses Neo4j hybrid retrieval (Document vector + Chunk fulltext) when Neo4j is configured. — Task 6.
  - [ ] Query traces show `neo4j_hybrid` vs `local_hybrid`. — Task 6.
  - [ ] Tests pass without Neo4j or OpenAI via the deterministic local fallback; no Customer-A/Vendor-X/SLA hardcoded boosts remain. — Tasks 3, 4.
  - [ ] Canned Customer-A/Vendor-X answer branch removed; answers synthesized from retrieved evidence (OpenAI when configured, else extractive); "not answerable from available knowledge" on weak evidence. — Task 3.
