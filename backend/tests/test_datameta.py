from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.datameta import DataMetaService
from app.main import app, mcp_tools


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


if __name__ == "__main__":
    unittest.main()
