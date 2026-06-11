"""MCP tool-call observability.

Records one telemetry row per MCP tool invocation — tool name, caller,
latency, and success/error — so the server's behaviour is inspectable rather
than opaque. This is the runtime counterpart to the static contract tests in
tests/test_mcp_contract.py: the tests pin the tool *surface*, this records the
tool *traffic*.

Telemetry is in-memory (a bounded ring) so it is zero-config and never grows
without bound. If DATAMETA_TELEMETRY_LOG points at a path, each row is also
appended there as JSON Lines for durable audit.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, deque
from threading import Lock
from typing import Any

# Tools that mutate state (kept in sync with the destructive/write tools in
# main.mcp_tools()). Used to tag telemetry rows so write traffic is easy to
# audit separately from reads.
WRITE_TOOLS = {
    "datameta_index_repos",
    "datameta_author_proposal",
    "datameta_commit_proposal",
    "datameta_flag_outlier",
    "datameta_resolve_flag",
}

_MAX_ROWS = 1000


class ToolCallRecorder:
    """Thread-safe bounded recorder of MCP tool-call telemetry."""

    def __init__(self, maxlen: int = _MAX_ROWS) -> None:
        self._rows: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = Lock()
        self._seq = 0

    def record(
        self,
        *,
        tool: str,
        user_id: str | None,
        latency_ms: float,
        ok: bool,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        self._seq += 1
        row = {
            "seq": self._seq,
            "tool": tool,
            "user_id": user_id,
            "write": tool in WRITE_TOOLS,
            "latency_ms": round(latency_ms, 1),
            "ok": ok,
            "error_type": error_type,
        }
        with self._lock:
            self._rows.append(row)
        log_path = os.environ.get("DATAMETA_TELEMETRY_LOG")
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
            except OSError:
                pass  # telemetry must never break a tool call
        return row

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._rows)
        return rows[-limit:][::-1]  # newest first

    def summary(self) -> dict[str, Any]:
        with self._lock:
            rows = list(self._rows)
        total = len(rows)
        if total == 0:
            return {"total_calls": 0, "error_rate": 0.0, "per_tool": {}}

        errors = sum(1 for r in rows if not r["ok"])
        latencies = sorted(r["latency_ms"] for r in rows)
        per_tool_count = Counter(r["tool"] for r in rows)
        per_tool_err = Counter(r["tool"] for r in rows if not r["ok"])

        def pct(p: float) -> float:
            idx = max(0, min(len(latencies) - 1, int(round(p * len(latencies))) - 1))
            return latencies[idx]

        return {
            "total_calls": total,
            "error_count": errors,
            "error_rate": round(errors / total, 4),
            "write_calls": sum(1 for r in rows if r["write"]),
            "latency_p50_ms": pct(0.50),
            "latency_p95_ms": pct(0.95),
            "per_tool": {
                tool: {
                    "calls": count,
                    "errors": per_tool_err.get(tool, 0),
                }
                for tool, count in sorted(per_tool_count.items())
            },
        }


recorder = ToolCallRecorder()


class instrument:
    """Context manager timing one tool call and recording its telemetry.

    Records on both success and failure (re-raising the exception so the
    caller's error handling is unchanged)::

        with instrument(name, arguments):
            payload = dispatch_tool(name, arguments)
    """

    def __init__(self, tool: str, arguments: dict[str, Any]) -> None:
        self.tool = tool
        self.user_id = (arguments or {}).get("user_id")
        self._start = 0.0

    def __enter__(self) -> "instrument":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        latency_ms = (time.perf_counter() - self._start) * 1000.0
        recorder.record(
            tool=self.tool,
            user_id=self.user_id,
            latency_ms=latency_ms,
            ok=exc_type is None,
            error_type=exc_type.__name__ if exc_type else None,
        )
        return False  # never suppress
