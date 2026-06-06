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
            "id": "arr-finance-board",
            "type": "definition",
            "entity": "ARR",
            "scope": "board_reporting",
            "team": "finance",
            "title": "ARR for board reporting",
            "summary": "Finance ARR uses active MRR multiplied by 10 for board reporting.",
            "formula_sql": "SELECT ROUND(SUM(monthly_recurring_revenue) * 10, 2) AS arr FROM {table} WHERE status = 'active'",
            "required_columns": ["monthly_recurring_revenue", "status"],
            "preferred_tables": ["subscriptions"],
            "path": "finance/arr.md",
            "body_markdown": "Finance now defines ARR as active MRR multiplied by 10 for board reporting.",
            "search_terms": ["arr", "finance", "board reporting", "active mrr"],
            "neo4j_labels": ["Document", "Definition"],
            "neo4j_relationships": [{"type": "DEFINES", "target": "ARR"}],
        }


class DataMetaServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.service = DataMetaService(Path(self.temp_dir.name))
        self.service.ensure_ready()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prepare_arr_prompts_between_finance_and_renewals(self) -> None:
        result = self.service.prepare_calculation("How to calculate ARR?", "junior.analyst")
        teams = {definition["team"] for definition in result["definitions"]}
        self.assertTrue(result["requires_choice"])
        self.assertEqual({"finance", "renewals"}, teams)

    def test_bootstrap_model_config_reflects_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "",
                "OPENAI_MODEL": "",
                "DATAMETA_REASONING_MODEL": "",
                "OPENAI_EMBEDDING_MODEL": "",
                "DATAMETA_EMBEDDING_MODEL": "",
            },
        ):
            local = self.service.bootstrap("junior.analyst")["models"]
        self.assertEqual(local["mode"], "local_deterministic")
        self.assertFalse(local["api_key_configured"])
        self.assertIsNone(local["reasoning"])
        self.assertIsNone(local["embedding"])

        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "",
                "DATAMETA_REASONING_MODEL": "configured-reasoning",
                "OPENAI_EMBEDDING_MODEL": "",
                "DATAMETA_EMBEDDING_MODEL": "configured-embedding",
            },
        ):
            configured = self.service.bootstrap("junior.analyst")["models"]
        self.assertEqual(configured["mode"], "openai_configured")
        self.assertTrue(configured["api_key_configured"])
        self.assertEqual(configured["reasoning"], "configured-reasoning")
        self.assertEqual(configured["embedding"], "configured-embedding")

        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "",
                "DATAMETA_REASONING_MODEL": "",
            },
        ):
            missing_model = self.service.bootstrap("junior.analyst")["models"]
        self.assertEqual(missing_model["mode"], "openai_missing_model")
        self.assertTrue(missing_model["api_key_configured"])
        self.assertIsNone(missing_model["reasoning"])

    def test_ask_answers_arr_definition_question(self) -> None:
        result = self.service.ask("junior.analyst", "What is the definition of ARR?")
        self.assertEqual(result["intent"], "answer")
        self.assertEqual(result["action"], "datameta_answer")
        self.assertIn("ARR definitions", result["answer"])
        self.assertTrue(result["calculation_prompt"]["requires_choice"])

    def test_ask_suspicious_data_prompt_requests_missing_detail(self) -> None:
        result = self.service.ask("olivia.ops", "Data point seems suspicious")
        self.assertEqual(result["intent"], "flag_outlier")
        self.assertEqual(result["action"], "needs_more_detail")
        self.assertTrue(result["needs_more_detail"])
        self.assertIn("table_name", result["missing"])
        self.assertIn("subject", result["missing"])
        self.assertIn("description", result["missing"])

    def test_ask_flags_detailed_suspicious_data_prompt(self) -> None:
        result = self.service.ask(
            "olivia.ops",
            "orders ord_005 May 4 GMV spike looks suspicious because GMV is more than 10x the prior three days.",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "datameta_flag_outlier")
        self.assertEqual(result["flag"]["status"], "pending_owner_review")

    def test_mcp_exposes_plain_language_ask_tool(self) -> None:
        tools = {tool["name"]: tool for tool in mcp_tools()}
        self.assertIn("datameta_ask", tools)
        self.assertEqual(["question"], tools["datameta_ask"]["inputSchema"]["required"])

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

    def test_run_finance_arr_calculation(self) -> None:
        result = self.service.run_calculation("junior.analyst", "arr-finance-board", "subscriptions")
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["value"], 30240.0)

    def test_rbac_blocks_inaccessible_table(self) -> None:
        with self.assertRaises(PermissionError):
            self.service.run_calculation("ravi.renewals", "arr-finance-board", "subscriptions")

    def test_outlier_requires_owner_resolution(self) -> None:
        flag = self.service.flag_outlier(
            "olivia.ops",
            "orders",
            "May 4 GMV spike",
            "GMV is more than 10x the previous three days.",
        )
        with self.assertRaises(PermissionError):
            self.service.resolve_flag("olivia.ops", flag["flag_id"], "Looks valid.")
        resolved = self.service.resolve_flag("dina.data", flag["flag_id"], "The marketplace feed duplicated a single order.")
        self.assertTrue(resolved["ok"])
        self.assertIn("commit_hash", resolved)

    def test_pipeline_returns_chart_and_table(self) -> None:
        run = self.service.run_pipeline("junior.analyst")
        self.assertTrue(run["ok"])
        self.assertGreater(len(run["output"]["table"]), 0)
        self.assertGreater(len(run["output"]["chart"]), 0)

    def test_authoring_conflict_requires_confirmation(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "", "OPENAI_MODEL": "", "DATAMETA_REASONING_MODEL": ""}):
            proposal = self.service.create_author_proposal(
                "maya.finance",
                "Finance definition of ARR for board reporting is active MRR times 12.",
                "finance",
            )
        validation = proposal["validation"]
        self.assertTrue(validation["needs_confirmation"])
        commit = self.service.commit_proposal("maya.finance", proposal["proposal_id"], confirm_overwrite=False)
        self.assertFalse(commit["ok"])
        committed = self.service.commit_proposal("maya.finance", proposal["proposal_id"], confirm_overwrite=True)
        self.assertTrue(committed["ok"])
        self.assertTrue(committed["commit_hash"])

    def test_openai_authoring_generates_markdown_graph_and_search_metadata(self) -> None:
        self.service.openai_client = FakeOpenAIClient()
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "DATAMETA_REASONING_MODEL": "test-authoring-model",
                "OPENAI_MODEL": "",
            },
        ):
            proposal = self.service.create_author_proposal(
                "maya.finance",
                "Change ARR definition for board reporting to active MRR times 10.",
                "finance",
            )
        self.assertEqual(proposal["authoring_source"], "openai_responses")
        self.assertEqual(proposal["model"], "test-authoring-model")
        self.assertIn("active MRR multiplied by 10", proposal["markdown"])
        self.assertIn("search_terms: arr,finance,board reporting,active mrr", proposal["markdown"])
        self.assertIn('"target": "ARR"', proposal["markdown"])
        self.assertTrue(proposal["validation"]["needs_confirmation"])


if __name__ == "__main__":
    unittest.main()
