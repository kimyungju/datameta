from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.datameta import DataMetaService
from app.main import app, mcp_tools


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def structured_json(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {
            "id": "customer-a-sla-vendor-x-credit-review",
            "type": "policy",
            "entity": "Customer A Availability SLA",
            "scope": "customer_a_availability",
            "team": "legal-contracts",
            "title": "Customer A Vendor X SLA Credit Review",
            "summary": "Customer A SLA updates require Vendor X evidence before credit language is approved.",
            "formula_sql": "",
            "required_columns": [],
            "preferred_tables": [],
            "path": "legal-contracts/proposed-updates/customer-a-vendor-x-sla-credit-review.md",
            "body_markdown": "Legal must review Vendor X evidence before Customer A service-credit language is approved.",
            "search_terms": ["Customer A", "Vendor X", "SLA", "service credit"],
            "neo4j_labels": ["Document", "Policy"],
            "neo4j_relationships": [{"type": "APPLIES_TO", "target": "Customer A"}],
        }

class DataMetaServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "",
                "DATAMETA_EMBEDDING_PROVIDER": "local",
                "DATAMETA_STRICT_OPENAI_EMBEDDINGS": "",
            },
        )
        self.env_patch.start()
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.service = DataMetaService(Path(self.temp_dir.name))
        self.service.ensure_ready()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        self.env_patch.stop()

    def test_seed_corpus_contains_required_repos_and_canonical_docs(self) -> None:
        inventory = self.service.repo_inventory()
        repos = {repo["repository"] for repo in inventory["repositories"]}
        self.assertEqual(
            {
                "security-incident-response",
                "legal-contracts",
                "customer-success-ops",
                "platform-operations",
                "vendor-risk-management",
                "data-governance",
                "finance",
                "renewals",
                "sales",
            },
            repos,
        )
        paths = {doc["path"] for doc in self.service.all_documents()}
        self.assertIn("legal-contracts/customer-agreements/customer-a-availability-sla.md", paths)
        self.assertIn("platform-operations/slo-measurement/customer-a-may-2026-availability.md", paths)
        self.assertIn("vendor-risk-management/vendor-x/vendor-x-risk-register.md", paths)

    def test_all_seeded_namespace_nodes_have_required_metadata(self) -> None:
        index = self.service.index_repos()
        self.assertTrue(index["ok"])
        self.assertEqual([], index["metadata_completeness"])
        self.assertEqual(9, index["counts"]["repositories"])
        self.assertEqual(21, index["counts"]["folders"])
        self.assertEqual(63, index["counts"]["files"])
        self.assertEqual(71, index["counts"]["chunks"])
        self.assertGreaterEqual(index["counts"]["customers"], 3)
        self.assertGreaterEqual(index["counts"]["vendors"], 2)
        self.assertGreaterEqual(index["counts"]["incidents"], 2)
        self.assertTrue(index["neo4j"]["required"])

    def test_customer_a_vendor_x_sla_query_routes_to_expected_repositories(self) -> None:
        result = self.service.multirepo_query(
            "junior.analyst",
            "Customer A says we missed their availability SLA during the Vendor X incident. What should we do?",
            True,
        )
        repos = {repo["repository"] for repo in result["shortlisted_repositories"]}
        self.assertEqual(4, len(repos))
        # The demo-critical teams: SLA terms (legal), measured availability
        # (platform-operations), and complaint/escalation handling (customer
        # success) must be routed to. The fourth slot is corpus-dependent.
        self.assertIn("legal-contracts", repos)
        self.assertIn("platform-operations", repos)
        self.assertIn("customer-success-ops", repos)
        self.assertTrue(result["answerable"])
        self.assertIn("99.62 percent", result["answer"])
        cited_paths = {citation["file_path"] for citation in result["citations"]}
        self.assertIn("legal-contracts/customer-agreements/customer-a-availability-sla.md", cited_paths)
        self.assertIn("platform-operations/slo-measurement/customer-a-may-2026-availability.md", cited_paths)

    def test_customer_a_sla_ranks_above_customer_b_and_c_distractors(self) -> None:
        result = self.service.multirepo_query(
            "junior.analyst",
            "Customer A availability SLA Vendor X",
            True,
        )
        agreement_files = [
            item["path"]
            for item in result["shortlisted_files"]
            if item["path"].startswith("legal-contracts/customer-agreements/")
        ]
        self.assertGreaterEqual(len(agreement_files), 3)
        self.assertEqual("legal-contracts/customer-agreements/customer-a-availability-sla.md", agreement_files[0])

    def test_vendor_x_files_rank_and_cite_above_vendor_y_distractors(self) -> None:
        result = self.service.multirepo_query(
            "junior.analyst",
            "Vendor X incident evidence for Customer A SLA review",
            True,
        )
        vendor_files = [
            item["path"]
            for item in result["shortlisted_files"]
            if item["path"].startswith("vendor-risk-management/")
        ]
        self.assertTrue(vendor_files)
        self.assertTrue(vendor_files[0].startswith("vendor-risk-management/vendor-x/"))
        cited_paths = {citation["file_path"] for citation in result["citations"]}
        self.assertIn("vendor-risk-management/vendor-x/vendor-x-risk-register.md", cited_paths)
        self.assertNotIn("vendor-risk-management/vendor-y/vendor-y-export-delay.md", cited_paths)

    def test_unanswerable_query_does_not_invent_facts(self) -> None:
        result = self.service.multirepo_query("junior.analyst", "What is the Singapore pantry catering policy?", True)
        self.assertFalse(result["answerable"])
        self.assertIn("Not answerable from available knowledge", result["answer"])
        self.assertEqual([], result["citations"])

    def test_markdown_search_and_read_are_access_controlled(self) -> None:
        search = self.service.search_markdown_files("leah.legal", "Customer A SLA", 20)
        paths = {file["path"] for file in search["files"]}
        self.assertIn("legal-contracts/customer-agreements/customer-a-availability-sla.md", paths)
        self.assertNotIn("security-incident-response/triage-runbooks/vendor-outage-triage.md", paths)

        file = self.service.read_markdown_file("leah.legal", "legal-contracts/customer-agreements/customer-a-availability-sla.md")
        self.assertIn("Customer A's enterprise agreement", file["markdown"])
        with self.assertRaises(PermissionError):
            self.service.read_markdown_file("leah.legal", "security-incident-response/triage-runbooks/vendor-outage-triage.md")

    def test_folder_subagents_only_run_for_prefiltered_folders_and_read_after_file_selection(self) -> None:
        result = self.service.multirepo_query(
            "junior.analyst",
            "Customer A says we missed their availability SLA during the Vendor X incident. What should we do?",
            True,
        )
        shortlisted_folders = {folder["path"].removesuffix("/.datameta.md") for folder in result["shortlisted_folders"]}
        spawned = set(result["trace"]["folder_subagents_spawned"])
        self.assertEqual(shortlisted_folders, spawned)
        for folder_result in result["folder_subagent_findings"]:
            selected = {item["path"] for item in folder_result["selected_files"]}
            self.assertLessEqual(set(folder_result["full_content_files_read"]), selected)

    def test_final_answer_cites_actual_markdown_files(self) -> None:
        result = self.service.multirepo_query(
            "junior.analyst",
            "Customer A says we missed their availability SLA during the Vendor X incident. What should we do?",
            True,
        )
        cited_paths = {citation["file_path"] for citation in result["citations"]}
        self.assertIn("legal-contracts/customer-agreements/customer-a-availability-sla.md", cited_paths)
        self.assertIn("platform-operations/incidents/vendor-x-outage-2026-05-20.md", cited_paths)
        self.assertIn("platform-operations/slo-measurement/customer-a-may-2026-availability.md", cited_paths)
        self.assertIn("customer-success-ops/customer-a/customer-a-escalation-playbook.md", cited_paths)
        self.assertIn("data-governance/audit-evidence/customer-a-may-2026-evidence-pack.md", cited_paths)

    def test_mcp_exposes_multirepo_tools(self) -> None:
        tools = {tool["name"]: tool for tool in mcp_tools()}
        self.assertIn("datameta_index_repos", tools)
        self.assertIn("datameta_repo_inventory", tools)
        self.assertIn("datameta_multirepo_query", tools)
        self.assertIn("datameta_search_markdown", tools)
        self.assertIn("datameta_read_markdown_file", tools)
        self.assertEqual(["question"], tools["datameta_multirepo_query"]["inputSchema"]["required"])

    def test_mcp_lists_every_dispatchable_tool_and_no_pipeline(self) -> None:
        listed = {tool["name"] for tool in mcp_tools()}
        # The calculation/flag tools must be discoverable: datameta_ask's ARR
        # pause tells the agent to call datameta_run_calculation next.
        self.assertIn("datameta_prepare_calculation", listed)
        self.assertIn("datameta_run_calculation", listed)
        self.assertIn("datameta_flag_outlier", listed)
        self.assertIn("datameta_resolve_flag", listed)
        self.assertNotIn("datameta_run_pipeline", listed)

    def test_mcp_arr_conflict_flow_is_completable_end_to_end(self) -> None:
        with TestClient(app) as client:
            ask = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "datameta_ask",
                        "arguments": {"question": "Help me calculate ARR for ASEAN", "user_id": "ops.associate"},
                    },
                },
            )
            self.assertEqual(200, ask.status_code)
            pause = ask.json()["result"]["structuredContent"]
            self.assertEqual("choose_definition", pause["intent"])
            listed = {tool["name"] for tool in mcp_tools()}
            self.assertIn(pause["action"], listed)
            definition = next(d for d in pause["definitions"] if d["id"] == "arr-finance-board")
            run = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": pause["action"],
                        "arguments": {
                            "definition_id": definition["id"],
                            "table": definition["accessible_tables"][0]["table"],
                            "user_id": "ops.associate",
                        },
                    },
                },
            )
            self.assertEqual(200, run.status_code)
            calculation = run.json()["result"]["structuredContent"]
            self.assertEqual(684000.0, calculation["result"]["value"])

    def test_pipeline_endpoint_is_removed(self) -> None:
        with TestClient(app) as client:
            response = client.post("/api/pipeline/run", json={"runbook_id": "gmv-category-ranker"})
        self.assertEqual(404, response.status_code)

    def test_mcp_initialized_notification_returns_accepted_without_body(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.content, b"")

    def test_health_route_is_safe_for_deployment_smoke_tests(self) -> None:
        with TestClient(app) as client:
            root = client.get("/")
            health = client.get("/api/health")
        self.assertEqual(root.status_code, 200)
        self.assertEqual(health.status_code, 200)
        self.assertTrue(root.json()["ok"])
        self.assertTrue(health.json()["ok"])

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

    def test_sync_prunes_nodes_missing_from_markdown_source(self) -> None:
        self.service.index_repos()
        index = self.service._multirepo_index
        statements = self.service._build_neo4j_sync_statements(index)
        prune_statements = [s for s in statements if "DETACH DELETE" in s["statement"]]
        pruned_labels = {s["statement"].split("(n:")[1].split(")")[0] for s in prune_statements}
        self.assertEqual({"Repository", "Folder", "Document", "Chunk"}, pruned_labels)
        document_prune = next(s for s in prune_statements if "Document" in s["statement"])
        self.assertEqual(
            sorted(item["id"] for item in index["files"].values()),
            sorted(document_prune["parameters"]["ids"]),
        )

    def test_chunk_sync_never_writes_embedding(self) -> None:
        self.service.index_repos()
        index = self.service._multirepo_index
        with patch.dict("os.environ", {"DATAMETA_NEO4J_VECTOR_DIMENSIONS": "256"}):
            statements = self.service._build_neo4j_sync_statements(index)
        chunk_statements = [s for s in statements if "MERGE (c:Chunk" in s["statement"]]
        self.assertTrue(chunk_statements)
        self.assertNotIn("c.embedding", chunk_statements[0]["statement"])

    def test_openai_authoring_still_generates_markdown_graph_and_search_metadata(self) -> None:
        self.service.openai_client = FakeOpenAIClient()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "DATAMETA_REASONING_MODEL": "test-authoring-model"}):
            proposal = self.service.create_author_proposal(
                "leah.legal",
                "Change Customer A SLA handling so Vendor X evidence is required before credit language is approved.",
                "legal-contracts",
            )
        self.assertEqual(proposal["authoring_source"], "openai_responses")
        self.assertEqual(proposal["model"], "test-authoring-model")
        self.assertIn("Vendor X evidence", proposal["markdown"])
        self.assertIn("search_terms: Customer A,Vendor X,SLA,service credit", proposal["markdown"])
        self.assertIn('"target": "Customer A"', proposal["markdown"])

    def test_conflict_check_includes_prior_committed_markdown(self) -> None:
        self.service.openai_client = FakeOpenAIClient()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "DATAMETA_REASONING_MODEL": "test-authoring-model"}):
            proposal = self.service.create_author_proposal(
                "leah.legal",
                "Change Customer A SLA handling so Vendor X evidence is required before credit language is approved.",
                "legal-contracts",
            )
        validation = self.service.validate_proposal(proposal["proposal_id"], "leah.legal")
        conflict_check = next(
            check for check in validation["checks"] if check["name"] == "Same entity + scope conflict check"
        )
        self.assertFalse(conflict_check["ok"])
        self.assertTrue(validation["needs_confirmation"])
        conflicts = conflict_check["conflicts"]
        self.assertTrue(conflicts)
        existing = self.service.read_markdown_file(
            "leah.legal", "legal-contracts/customer-agreements/customer-a-availability-sla.md"
        )
        prior = next(
            conflict
            for conflict in conflicts
            if conflict["path"] == "legal-contracts/customer-agreements/customer-a-availability-sla.md"
        )
        self.assertEqual("Customer A Availability SLA", prior["entity"])
        self.assertEqual("customer_a_availability", prior["scope"])
        self.assertIn("markdown", prior)
        self.assertEqual(existing["markdown"], prior["markdown"])
        self.assertIn("Customer A's enterprise agreement", prior["markdown"])


    def test_ask_arr_conflict_pauses_and_each_definition_computes_distinct_value(self) -> None:
        result = self.service.ask("ops.associate", "Help me calculate ARR for ASEAN")
        self.assertEqual("choose_definition", result["intent"])
        self.assertTrue(result["requires_choice"])
        definition_ids = {definition["id"] for definition in result["definitions"]}
        self.assertEqual(
            {"arr-finance-board", "arr-renewals-forecast", "arr-sales-presentation"},
            definition_ids,
        )
        expected_values = {
            "arr-finance-board": 684000.0,
            "arr-renewals-forecast": 700000.0,
            "arr-sales-presentation": 790000.0,
        }
        for definition in result["definitions"]:
            table = definition["accessible_tables"][0]["table"]
            calculation = self.service.run_calculation("ops.associate", definition["id"], table)
            self.assertEqual(expected_values[definition["id"]], calculation["result"]["value"])

    def test_ask_arr_without_conflicting_visibility_does_not_pause(self) -> None:
        result = self.service.ask("leah.legal", "Help me calculate ARR for ASEAN")
        self.assertNotEqual("choose_definition", result["intent"])

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
        doc_path = "platform-operations/incidents/vendor-x-outage-2026-05-20.md"
        vector_rows = [
            {
                "document_id": "inc-vendor-x-2026-05-20",
                "path": doc_path,
                "title": "Vendor X Availability Incident on 2026-05-20",
                "summary": "Vendor X token validation degradation.",
                "team": "platform-operations",
                "score": 0.92,
            },
            {
                # Distractor: a Customer C document must be filtered out of the
                # evidence for a Customer A query by the no-guessing filter.
                "document_id": "customer-c-availability-sla",
                "path": "legal-contracts/customer-agreements/customer-c-availability-sla.md",
                "title": "Customer C Availability SLA",
                "summary": "Customer C has a 99.90 percent availability target.",
                "team": "legal-contracts",
                "score": 0.88,
            },
        ]
        chunk_rows = [
            {
                "document_id": "inc-vendor-x-2026-05-20",
                "path": doc_path,
                "title": "Vendor X Availability Incident on 2026-05-20",
                "summary": "Vendor X token validation degradation.",
                "heading": "Timeline",
                "text": "Vendor X ... Customer A.",
                "team": "platform-operations",
                "score": 3.1,
            }
        ]

        def fake(statement, parameters=None):
            self.assertIn("admin", parameters)
            self.assertIn("teams", parameters)
            if "VECTOR INDEX datameta_document_embedding" in statement:
                return vector_rows
            if "db.index.fulltext.queryNodes" in statement:
                return chunk_rows
            return []

        with patch.dict(
            "os.environ",
            {
                "DATAMETA_NEO4J_URL": "bolt://x",
                "DATAMETA_NEO4J_USER": "neo4j",
                "DATAMETA_NEO4J_PASSWORD": "secret",
                "DATAMETA_NEO4J_VECTOR_DIMENSIONS": "256",
            },
        ):
            with patch.object(self.service, "_neo4j_read", side_effect=fake):
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
        self.assertEqual("Timeline", item["heading"])
        cited = {c["file_path"] for c in result["citations"]}
        self.assertIn(doc_path, cited)

    def test_neo4j_hybrid_respects_rbac(self) -> None:
        captured: dict[str, object] = {}

        def capture(statement, parameters=None):
            captured["teams"] = parameters.get("teams")
            captured["admin"] = parameters.get("admin")
            return []

        with patch.dict(
            "os.environ",
            {
                "DATAMETA_NEO4J_URL": "bolt://x",
                "DATAMETA_NEO4J_USER": "neo4j",
                "DATAMETA_NEO4J_PASSWORD": "secret",
                "DATAMETA_NEO4J_VECTOR_DIMENSIONS": "256",
            },
        ):
            with patch.object(self.service, "_neo4j_read", side_effect=capture):
                self.service.multirepo_query("leah.legal", "Customer A SLA", True)
        self.assertFalse(captured["admin"])
        self.assertNotIn("security-incident-response", captured["teams"])
        self.assertIn("legal-contracts", captured["teams"])


if __name__ == "__main__":
    unittest.main()
