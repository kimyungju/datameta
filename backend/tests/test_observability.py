"""Tests for MCP tool-call observability."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app
from app.observability import ToolCallRecorder, WRITE_TOOLS, instrument, recorder


class RecorderTest(unittest.TestCase):
    def test_records_and_summarizes(self) -> None:
        r = ToolCallRecorder()
        r.record(tool="datameta_ask", user_id="u1", latency_ms=10.0, ok=True)
        r.record(tool="datameta_ask", user_id="u1", latency_ms=30.0, ok=True)
        r.record(tool="datameta_commit_proposal", user_id="u2", latency_ms=5.0,
                 ok=False, error_type="PermissionError")

        s = r.summary()
        self.assertEqual(3, s["total_calls"])
        self.assertEqual(1, s["error_count"])
        self.assertAlmostEqual(1 / 3, s["error_rate"], places=3)
        self.assertEqual(1, s["write_calls"])  # only commit_proposal is a write
        self.assertEqual(2, s["per_tool"]["datameta_ask"]["calls"])
        self.assertEqual(1, s["per_tool"]["datameta_commit_proposal"]["errors"])

    def test_empty_summary_is_safe(self) -> None:
        s = ToolCallRecorder().summary()
        self.assertEqual(0, s["total_calls"])
        self.assertEqual(0.0, s["error_rate"])
        self.assertEqual({}, s["per_tool"])

    def test_recent_is_newest_first_and_bounded(self) -> None:
        r = ToolCallRecorder(maxlen=3)
        for i in range(5):
            r.record(tool=f"t{i}", user_id=None, latency_ms=1.0, ok=True)
        recent = r.recent(limit=10)
        self.assertEqual(3, len(recent))  # ring bounded to 3
        self.assertEqual("t4", recent[0]["tool"])  # newest first

    def test_write_tools_tagged(self) -> None:
        r = ToolCallRecorder()
        r.record(tool="datameta_commit_proposal", user_id=None, latency_ms=1.0, ok=True)
        r.record(tool="datameta_retrieve", user_id=None, latency_ms=1.0, ok=True)
        rows = {row["tool"]: row["write"] for row in r.recent()}
        self.assertTrue(rows["datameta_commit_proposal"])
        self.assertFalse(rows["datameta_retrieve"])


class InstrumentTest(unittest.TestCase):
    def test_records_success(self) -> None:
        r = ToolCallRecorder()
        with _use_recorder(r):
            with instrument("datameta_ask", {"user_id": "alice"}):
                pass
        row = r.recent()[0]
        self.assertEqual("datameta_ask", row["tool"])
        self.assertEqual("alice", row["user_id"])
        self.assertTrue(row["ok"])

    def test_records_and_reraises_on_error(self) -> None:
        r = ToolCallRecorder()
        with _use_recorder(r):
            with self.assertRaises(ValueError):
                with instrument("datameta_run_calculation", {"user_id": "bob"}):
                    raise ValueError("boom")
        row = r.recent()[0]
        self.assertFalse(row["ok"])
        self.assertEqual("ValueError", row["error_type"])


class TelemetryEndpointTest(unittest.TestCase):
    def test_tool_call_is_recorded_and_exposed(self) -> None:
        with TestClient(app) as client:
            before = client.get("/api/telemetry").json()["summary"]["total_calls"]
            client.post("/mcp", json={
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "datameta_repo_inventory", "arguments": {}},
            })
            after = client.get("/api/telemetry").json()
            self.assertEqual(before + 1, after["summary"]["total_calls"])
            self.assertEqual("datameta_repo_inventory", after["recent"][0]["tool"])
            self.assertTrue(after["recent"][0]["ok"])

    def test_failed_tool_call_recorded_as_error(self) -> None:
        with TestClient(app) as client:
            client.post("/mcp", json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "datameta_unknown_tool", "arguments": {}},
            })
            recent = client.get("/api/telemetry").json()["recent"]
            self.assertEqual("datameta_unknown_tool", recent[0]["tool"])
            self.assertFalse(recent[0]["ok"])


def _use_recorder(temp):
    """Context manager swapping the module-global recorder for a temp one."""
    import app.observability as obs

    class _Swap:
        def __enter__(self):
            self.orig = obs.recorder
            obs.recorder = temp
            return temp

        def __exit__(self, *a):
            obs.recorder = self.orig
            return False

    return _Swap()


if __name__ == "__main__":
    unittest.main()
