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
        self.temp_dir = tempfile.TemporaryDirectory()
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
        self.assertEqual(6, index["counts"]["repositories"])
        self.assertEqual(18, index["counts"]["folders"])
        self.assertEqual(54, index["counts"]["files"])
        self.assertEqual(61, index["counts"]["chunks"])
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
        repos = [repo["repository"] for repo in result["shortlisted_repositories"]]
        self.assertEqual(
            {"vendor-risk-management", "customer-success-ops", "legal-contracts", "platform-operations"},
            set(repos),
        )
        self.assertTrue(result["answerable"])
        self.assertIn("99.72 percent", result["answer"])
        self.assertIn("99.90 percent", result["answer"])

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
        self.assertIn("platform-operations/incidents/vendor-x-2026-05-availability-incident.md", cited_paths)
        self.assertIn("platform-operations/slo-measurement/customer-a-may-2026-availability.md", cited_paths)
        self.assertIn("customer-success-ops/customer-a/customer-a-escalation-playbook.md", cited_paths)
        self.assertIn("vendor-risk-management/vendor-x/vendor-x-risk-register.md", cited_paths)

    def test_mcp_exposes_multirepo_tools(self) -> None:
        tools = {tool["name"]: tool for tool in mcp_tools()}
        self.assertIn("datameta_index_repos", tools)
        self.assertIn("datameta_repo_inventory", tools)
        self.assertIn("datameta_multirepo_query", tools)
        self.assertEqual(["question"], tools["datameta_multirepo_query"]["inputSchema"]["required"])

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


if __name__ == "__main__":
    unittest.main()
