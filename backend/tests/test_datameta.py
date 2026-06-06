from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.datameta import DataMetaService


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
