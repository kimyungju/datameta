"""MCP server contract tests.

Validates the protocol surface an external agent (Codex, Claude) relies on:
the advertised tool registry is consistent everywhere, every schema is a
well-formed JSON Schema, permission annotations are present and truthful,
the bearer-token gate works, and unknown tools fail as JSON-RPC tool errors
rather than 500s.
"""

from __future__ import annotations

import inspect
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import main as main_module
from app.main import app, dispatch_tool, mcp_tools


def dispatchable_tool_names() -> set[str]:
    """Tool names handled by dispatch_tool, extracted from its source."""
    source = inspect.getsource(dispatch_tool)
    return set(re.findall(r'name == "([a-z0-9_]+)"', source))


class McpRegistryConsistencyTest(unittest.TestCase):
    def test_every_listed_tool_is_dispatchable(self) -> None:
        listed = {tool["name"] for tool in mcp_tools()}
        self.assertEqual(set(), listed - dispatchable_tool_names(),
                         "tools advertised via tools/list but not dispatchable")

    def test_every_dispatchable_tool_is_listed(self) -> None:
        listed = {tool["name"] for tool in mcp_tools()}
        self.assertEqual(set(), dispatchable_tool_names() - listed,
                         "tools dispatchable but hidden from tools/list")

    def test_bootstrap_advertises_full_registry(self) -> None:
        """The bootstrap payload's mcp.tools list must match the real server."""
        listed = {tool["name"] for tool in mcp_tools()}
        advertised = set(main_module.service.mcp_tool_names())
        self.assertEqual(listed, advertised,
                         "bootstrap mcp.tools out of sync with /mcp tools/list")


class McpSchemaValidityTest(unittest.TestCase):
    def test_schemas_are_well_formed(self) -> None:
        for tool in mcp_tools():
            with self.subTest(tool=tool["name"]):
                schema = tool["inputSchema"]
                self.assertEqual("object", schema["type"])
                properties = schema.get("properties", {})
                self.assertIsInstance(properties, dict)
                for required_field in schema.get("required", []):
                    self.assertIn(required_field, properties,
                                  f"required field '{required_field}' missing from properties")

    def test_every_tool_declares_permission_annotations(self) -> None:
        for tool in mcp_tools():
            with self.subTest(tool=tool["name"]):
                self.assertIn("annotations", tool)
                self.assertIn("readOnlyHint", tool["annotations"])

    def test_commit_is_the_only_destructive_tool(self) -> None:
        destructive = {
            tool["name"]
            for tool in mcp_tools()
            if tool["annotations"].get("destructiveHint")
        }
        self.assertEqual({"datameta_commit_proposal"}, destructive)

    def test_read_only_tools_do_not_claim_write_hints(self) -> None:
        for tool in mcp_tools():
            if tool["annotations"].get("readOnlyHint"):
                with self.subTest(tool=tool["name"]):
                    self.assertFalse(tool["annotations"].get("destructiveHint", False))


class McpAuthTest(unittest.TestCase):
    INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}

    def test_missing_bearer_token_rejected_when_configured(self) -> None:
        with patch.dict(os.environ, {"DATAMETA_MCP_TOKEN": "secret-token"}):
            with TestClient(app) as client:
                response = client.post("/mcp", json=self.INITIALIZE)
                self.assertEqual(401, response.status_code)

    def test_wrong_bearer_token_rejected(self) -> None:
        with patch.dict(os.environ, {"DATAMETA_MCP_TOKEN": "secret-token"}):
            with TestClient(app) as client:
                response = client.post(
                    "/mcp", json=self.INITIALIZE,
                    headers={"Authorization": "Bearer wrong"})
                self.assertEqual(401, response.status_code)

    def test_correct_bearer_token_accepted(self) -> None:
        with patch.dict(os.environ, {"DATAMETA_MCP_TOKEN": "secret-token"}):
            with TestClient(app) as client:
                response = client.post(
                    "/mcp", json=self.INITIALIZE,
                    headers={"Authorization": "Bearer secret-token"})
                self.assertEqual(200, response.status_code)
                self.assertEqual("datameta", response.json()["result"]["serverInfo"]["name"])

    def test_no_token_configured_allows_local_dev(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "DATAMETA_MCP_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            with TestClient(app) as client:
                response = client.post("/mcp", json=self.INITIALIZE)
                self.assertEqual(200, response.status_code)


class McpErrorHandlingTest(unittest.TestCase):
    def test_unknown_tool_returns_tool_error_not_500(self) -> None:
        with TestClient(app) as client:
            response = client.post("/mcp", json={
                "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                "params": {"name": "datameta_nonexistent", "arguments": {}},
            })
            self.assertEqual(200, response.status_code)
            body = response.json()
            self.assertTrue(body["result"]["isError"])
            self.assertIn("Unknown DataMeta tool", body["result"]["content"][0]["text"])

    def test_unknown_method_returns_jsonrpc_error(self) -> None:
        with TestClient(app) as client:
            response = client.post("/mcp", json={
                "jsonrpc": "2.0", "id": 8, "method": "resources/list", "params": {},
            })
            self.assertEqual(200, response.status_code)
            self.assertEqual(-32601, response.json()["error"]["code"])

    def test_rbac_denial_surfaces_as_tool_error(self) -> None:
        """A user reading a file outside their teams gets a clean tool error."""
        with TestClient(app) as client:
            response = client.post("/mcp", json={
                "jsonrpc": "2.0", "id": 9, "method": "tools/call",
                "params": {
                    "name": "datameta_read_markdown_file",
                    "arguments": {"path": "../../etc/passwd.md", "user_id": "junior.analyst"},
                },
            })
            self.assertEqual(200, response.status_code)
            self.assertTrue(response.json()["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
