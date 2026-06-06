from __future__ import annotations

import json
import os
import re
import shlex
import sqlite3
import subprocess
import base64
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "note"


def tokenize(text: str) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9]+", text.lower()) if part}


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if value and value[0] in {"'", '"'}:
            try:
                parsed = shlex.split(value, posix=True)
                value = parsed[0] if parsed else ""
            except ValueError:
                value = value.strip("'\"")
        os.environ[key] = value


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_env_file(PROJECT_ROOT / ".env")
load_env_file(PROJECT_ROOT / "backend" / ".env")


def configured_env_value(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def model_config() -> dict[str, Any]:
    api_key_configured = configured_env_value("OPENAI_API_KEY") is not None
    reasoning = configured_env_value("DATAMETA_REASONING_MODEL") or configured_env_value("OPENAI_MODEL")
    embedding = configured_env_value("DATAMETA_EMBEDDING_MODEL") or configured_env_value("OPENAI_EMBEDDING_MODEL")
    openai_ready = api_key_configured and reasoning is not None
    mode = "openai_configured" if openai_ready else "openai_missing_model" if api_key_configured else "local_deterministic"
    return {
        "mode": mode,
        "api_key_configured": api_key_configured,
        "reasoning": reasoning,
        "embedding": embedding,
        "env": {
            "api_key": "OPENAI_API_KEY",
            "reasoning": "DATAMETA_REASONING_MODEL or OPENAI_MODEL",
            "embedding": "DATAMETA_EMBEDDING_MODEL or OPENAI_EMBEDDING_MODEL",
        },
    }


PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "type",
        "entity",
        "scope",
        "team",
        "title",
        "summary",
        "formula_sql",
        "required_columns",
        "preferred_tables",
        "path",
        "body_markdown",
        "search_terms",
        "neo4j_labels",
        "neo4j_relationships",
    ],
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string", "enum": ["definition", "policy", "runbook", "note"]},
        "entity": {"type": "string"},
        "scope": {"type": "string"},
        "team": {"type": "string"},
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "formula_sql": {"type": "string"},
        "required_columns": {"type": "array", "items": {"type": "string"}},
        "preferred_tables": {"type": "array", "items": {"type": "string"}},
        "path": {"type": "string"},
        "body_markdown": {"type": "string"},
        "search_terms": {"type": "array", "items": {"type": "string"}},
        "neo4j_labels": {"type": "array", "items": {"type": "string"}},
        "neo4j_relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "target"],
                "properties": {
                    "type": {"type": "string"},
                    "target": {"type": "string"},
                },
            },
        },
    },
}


class OpenAIResponsesClient:
    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or configured_env_value("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

    def structured_json(self, *, system: str, user: str, schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI Responses API failed: {error.code} {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"OpenAI Responses API could not be reached: {error.reason}") from error

        output_text = self._output_text(response_payload)
        try:
            return json.loads(output_text)
        except json.JSONDecodeError as error:
            raise RuntimeError("OpenAI response did not contain valid JSON structured output") from error

    def _output_text(self, response_payload: dict[str, Any]) -> str:
        if isinstance(response_payload.get("output_text"), str):
            return response_payload["output_text"]
        chunks: list[str] = []
        for item in response_payload.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    chunks.append(content["text"])
                if content.get("type") == "refusal":
                    raise RuntimeError(f"OpenAI refused the authoring request: {content.get('refusal', '')}")
        if not chunks:
            raise RuntimeError("OpenAI response did not include output_text")
        return "".join(chunks)


@dataclass(frozen=True)
class User:
    id: str
    name: str
    roles: tuple[str, ...]
    read_teams: tuple[str, ...]
    write_teams: tuple[str, ...]
    tables: tuple[str, ...]

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles

    def can_read_team(self, team: str) -> bool:
        return self.is_admin or team in self.read_teams

    def can_write_team(self, team: str) -> bool:
        return self.is_admin or team in self.write_teams

    def can_access_table(self, table: str) -> bool:
        return self.is_admin or table in self.tables


SEEDED_USERS: dict[str, User] = {
    "junior.analyst": User(
        id="junior.analyst",
        name="Jamie Tan",
        roles=("analyst",),
        read_teams=("finance", "renewals", "analytics"),
        write_teams=(),
        tables=("subscriptions", "renewals", "orders"),
    ),
    "maya.finance": User(
        id="maya.finance",
        name="Maya Finance",
        roles=("analyst", "author"),
        read_teams=("finance", "analytics"),
        write_teams=("finance",),
        tables=("subscriptions", "revenue_events", "orders"),
    ),
    "ravi.renewals": User(
        id="ravi.renewals",
        name="Ravi Renewals",
        roles=("analyst", "author"),
        read_teams=("renewals", "analytics"),
        write_teams=("renewals",),
        tables=("renewals", "subscriptions"),
    ),
    "olivia.ops": User(
        id="olivia.ops",
        name="Olivia Ops",
        roles=("analyst", "author"),
        read_teams=("ops", "analytics", "data-ownership"),
        write_teams=("ops",),
        tables=("orders", "ops_daily_metrics"),
    ),
    "dina.data": User(
        id="dina.data",
        name="Dina Data Owner",
        roles=("data_owner", "author"),
        read_teams=("data-ownership", "ops", "analytics", "finance", "renewals"),
        write_teams=("data-ownership",),
        tables=("orders", "ops_daily_metrics", "subscriptions", "renewals", "revenue_events"),
    ),
    "ada.admin": User(
        id="ada.admin",
        name="Ada Admin",
        roles=("admin",),
        read_teams=("finance", "renewals", "ops", "analytics", "data-ownership"),
        write_teams=("finance", "renewals", "ops", "analytics", "data-ownership"),
        tables=("subscriptions", "renewals", "orders", "ops_daily_metrics", "revenue_events"),
    ),
}


class DataMetaService:
    def __init__(self, runtime_dir: Path | None = None, openai_client: OpenAIResponsesClient | None = None) -> None:
        root = PROJECT_ROOT
        configured = os.environ.get("DATAMETA_RUNTIME_DIR")
        default_runtime = Path("/tmp/datameta-runtime") if os.environ.get("VERCEL") else root / "runtime"
        self.runtime_dir = Path(configured) if configured else runtime_dir or default_runtime
        self.knowledge_repo = self.runtime_dir / "shoppy-knowledge"
        self.app_db = self.runtime_dir / "datameta.sqlite"
        self.warehouse_db = self.runtime_dir / "shoppy-warehouse.sqlite"
        self.openai_client = openai_client
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._ready = False

    def ensure_ready(self) -> None:
        if self._ready:
            return
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._init_app_db()
        self._init_warehouse()
        self._init_knowledge_repo()
        self._ready = True

    def reset_for_tests(self) -> None:
        self._ready = False
        if self.app_db.exists():
            self.app_db.unlink()
        if self.warehouse_db.exists():
            self.warehouse_db.unlink()
        if self.knowledge_repo.exists():
            for path in sorted(self.knowledge_repo.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                else:
                    path.rmdir()
            self.knowledge_repo.rmdir()
        self.ensure_ready()

    def users(self) -> list[dict[str, Any]]:
        self.ensure_ready()
        return [
            {
                "id": user.id,
                "name": user.name,
                "roles": list(user.roles),
                "read_teams": list(user.read_teams),
                "write_teams": list(user.write_teams),
                "tables": list(user.tables),
            }
            for user in SEEDED_USERS.values()
        ]

    def get_user(self, user_id: str | None) -> User:
        self.ensure_ready()
        if not user_id:
            return SEEDED_USERS["junior.analyst"]
        if user_id not in SEEDED_USERS:
            raise ValueError(f"Unknown seeded user: {user_id}")
        return SEEDED_USERS[user_id]

    def _connect_app(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.app_db)
        connection.row_factory = sqlite3.Row
        return connection

    def _connect_warehouse(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.warehouse_db)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_app_db(self) -> None:
        with self._connect_app() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS proposals (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    natural_language TEXT NOT NULL,
                    target_team TEXT NOT NULL,
                    target_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS outlier_flags (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    owner_team TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    resolved_at TEXT,
                    resolved_by TEXT,
                    resolution TEXT
                );

                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    runbook_id TEXT NOT NULL,
                    variant TEXT,
                    output_json TEXT NOT NULL
                );
                """
            )

    def _init_warehouse(self) -> None:
        if self.warehouse_db.exists():
            return
        with self._connect_warehouse() as db:
            db.executescript(
                """
                CREATE TABLE subscriptions (
                    subscription_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    plan TEXT NOT NULL,
                    monthly_recurring_revenue REAL NOT NULL,
                    status TEXT NOT NULL,
                    start_date TEXT NOT NULL
                );

                CREATE TABLE renewals (
                    renewal_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    renewal_arr REAL NOT NULL,
                    renewal_status TEXT NOT NULL,
                    renewal_date TEXT NOT NULL,
                    owner_team TEXT NOT NULL
                );

                CREATE TABLE orders (
                    order_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    gross_merchandise_value REAL NOT NULL,
                    refund_amount REAL NOT NULL,
                    order_date TEXT NOT NULL,
                    channel TEXT NOT NULL
                );

                CREATE TABLE ops_daily_metrics (
                    metric_date TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    owner_team TEXT NOT NULL
                );

                CREATE TABLE revenue_events (
                    event_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    mrr REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    event_date TEXT NOT NULL
                );
                """
            )
            db.executemany(
                "INSERT INTO subscriptions VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("sub_001", "cust_001", "Pro", 220.0, "active", "2026-01-15"),
                    ("sub_002", "cust_002", "Enterprise", 780.0, "active", "2026-02-01"),
                    ("sub_003", "cust_003", "Starter", 80.0, "paused", "2026-03-03"),
                    ("sub_004", "cust_004", "Pro", 300.0, "active", "2026-03-18"),
                    ("sub_005", "cust_005", "Enterprise", 1220.0, "active", "2026-04-09"),
                ],
            )
            db.executemany(
                "INSERT INTO renewals VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("ren_001", "cust_001", 2880.0, "committed", "2026-07-01", "renewals"),
                    ("ren_002", "cust_002", 9600.0, "likely", "2026-07-15", "renewals"),
                    ("ren_003", "cust_003", 900.0, "at_risk", "2026-08-01", "renewals"),
                    ("ren_004", "cust_005", 15120.0, "committed", "2026-08-20", "renewals"),
                ],
            )
            db.executemany(
                "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    ("ord_001", "cust_001", "Beauty", 4100.0, 100.0, "2026-05-01", "paid_search"),
                    ("ord_002", "cust_002", "Electronics", 8500.0, 250.0, "2026-05-02", "affiliate"),
                    ("ord_003", "cust_003", "Home", 1900.0, 0.0, "2026-05-02", "organic"),
                    ("ord_004", "cust_004", "Fashion", 7600.0, 1100.0, "2026-05-03", "paid_social"),
                    ("ord_005", "cust_005", "Electronics", 920000.0, 0.0, "2026-05-04", "marketplace_feed"),
                    ("ord_006", "cust_006", "Beauty", 5100.0, 300.0, "2026-05-04", "organic"),
                    ("ord_007", "cust_007", "Home", 4200.0, 200.0, "2026-05-05", "paid_search"),
                    ("ord_008", "cust_008", "Fashion", 6500.0, 500.0, "2026-05-05", "affiliate"),
                ],
            )
            db.executemany(
                "INSERT INTO ops_daily_metrics VALUES (?, ?, ?, ?)",
                [
                    ("2026-05-01", "gmv", 29500.0, "data-ownership"),
                    ("2026-05-02", "gmv", 33200.0, "data-ownership"),
                    ("2026-05-03", "gmv", 30100.0, "data-ownership"),
                    ("2026-05-04", "gmv", 945700.0, "data-ownership"),
                ],
            )
            db.executemany(
                "INSERT INTO revenue_events VALUES (?, ?, ?, ?, ?)",
                [
                    ("evt_001", "cust_001", 220.0, "new", "2026-01-15"),
                    ("evt_002", "cust_002", 780.0, "new", "2026-02-01"),
                    ("evt_003", "cust_004", 300.0, "expansion", "2026-03-18"),
                    ("evt_004", "cust_005", 1220.0, "new", "2026-04-09"),
                ],
            )

    def _init_knowledge_repo(self) -> None:
        self.knowledge_repo.mkdir(parents=True, exist_ok=True)
        if not (self.knowledge_repo / ".git").exists():
            try:
                self._git(["init"])
                self._git(["config", "user.name", "DataMeta Seed"])
                self._git(["config", "user.email", "seed@datameta.local"])
            except RuntimeError:
                if not os.environ.get("VERCEL"):
                    raise
        seed_files = self._seed_markdown_files()
        changed = False
        for relative_path, content in seed_files.items():
            path = self.knowledge_repo / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(content, encoding="utf-8")
                changed = True
        if changed:
            self._git(["add", "."], check=False)
            self._git_commit("Seed Shoppy knowledge base", "DataMeta Seed", "seed@datameta.local")

    def _seed_markdown_files(self) -> dict[str, str]:
        return {
            "finance/arr.md": """---
id: arr-finance-board
type: definition
entity: ARR
scope: board_reporting
team: finance
title: ARR for board reporting
summary: Finance ARR is active subscription MRR multiplied by 12.
formula_sql: SELECT ROUND(SUM(monthly_recurring_revenue) * 12, 2) AS arr FROM {table} WHERE status = 'active'
required_columns: monthly_recurring_revenue,status
preferred_tables: subscriptions
updated_by: Maya Finance
updated_at: 2026-06-04T09:00:00+00:00
---
# ARR for board reporting

Finance uses active subscription MRR multiplied by 12 for board reporting.

Use this definition when preparing the company board deck, investor updates, or financial close analysis.

Calculation:

`ARR = SUM(monthly_recurring_revenue where status = active) * 12`
""",
            "renewals/arr.md": """---
id: arr-renewals-forecast
type: definition
entity: ARR
scope: renewal_forecasting
team: renewals
title: ARR for renewal forecasting
summary: Renewals ARR uses committed and likely renewal ARR values for upcoming renewal periods.
formula_sql: SELECT ROUND(SUM(renewal_arr), 2) AS arr FROM {table} WHERE renewal_status IN ('committed', 'likely')
required_columns: renewal_arr,renewal_status
preferred_tables: renewals
updated_by: Ravi Renewals
updated_at: 2026-06-04T10:00:00+00:00
---
# ARR for renewal forecasting

Renewals uses already annualized renewal opportunity values for committed and likely renewals.

Use this definition when forecasting renewal coverage, renewal pipeline, or customer success operating reviews.

Calculation:

`ARR = SUM(renewal_arr where renewal_status is committed or likely)`
""",
            "ops/gmv-outlier-playbook.md": """---
id: gmv-outlier-playbook
type: policy
entity: GMV Outlier
scope: operations_quality
team: ops
title: GMV outlier escalation
summary: Ops can flag GMV spikes, but data-ownership resolves official data-quality status.
updated_by: Olivia Ops
updated_at: 2026-06-04T11:00:00+00:00
---
# GMV outlier escalation

Ops analysts may flag suspected GMV spikes when daily GMV is more than 10x the trailing three-day median.

The data-ownership team owns confirmation, rejection, or annotation before the issue becomes official knowledge.
""",
            "analytics/gmv-category-ranker.md": """---
id: gmv-category-ranker
type: runbook
entity: GMV Category Ranking
scope: weekly_ops_review
team: analytics
title: Weekly GMV category ranking
summary: Rank categories by net GMV using Shoppy colors and exclude refunded value.
required_tables: orders
chart_colors: #2563eb,#16a34a,#f97316,#db2777,#7c3aed
updated_by: Jamie Tan
updated_at: 2026-06-04T12:00:00+00:00
---
# Weekly GMV category ranking

Steps:

1. Read `orders`.
2. Compute `net_gmv = gross_merchandise_value - refund_amount`.
3. Group by category.
4. Rank by net GMV descending.
5. Use Shoppy category colors in this order: blue, green, orange, pink, violet.

Default chart: horizontal bar chart with category and net GMV.
""",
            "data-ownership/orders-owner.md": """---
id: orders-owner
type: policy
entity: Orders
scope: data_ownership
team: data-ownership
title: Orders table ownership
summary: The data-ownership team owns official quality notes for orders and ops_daily_metrics.
updated_by: Dina Data Owner
updated_at: 2026-06-04T13:00:00+00:00
---
# Orders table ownership

The data-ownership team owns official quality status for `orders` and `ops_daily_metrics`.

Other teams may flag suspected issues, but resolution belongs to data-ownership.
""",
        }

    def _git(self, args: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self.knowledge_repo,
                check=check,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError:
            if check:
                raise RuntimeError("git is not available in this runtime")
            return subprocess.CompletedProcess(["git", *args], 127, "", "git is not available in this runtime")

    def _git_commit(self, message: str, author_name: str, author_email: str) -> str | None:
        env = os.environ.copy()
        env.update(
            {
                "GIT_AUTHOR_NAME": author_name,
                "GIT_AUTHOR_EMAIL": author_email,
                "GIT_COMMITTER_NAME": author_name,
                "GIT_COMMITTER_EMAIL": author_email,
            }
        )
        result = self._git(["commit", "-m", message], check=False, env=env)
        if result.returncode != 0:
            if "nothing to commit" in result.stdout.lower() or "nothing to commit" in result.stderr.lower():
                return None
            if os.environ.get("VERCEL"):
                return f"serverless-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
            raise RuntimeError(result.stderr or result.stdout)
        rev_parse = self._git(["rev-parse", "HEAD"], check=False)
        if rev_parse.returncode != 0:
            return f"serverless-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        return rev_parse.stdout.strip()

    def parse_document(self, path: Path) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        metadata: dict[str, str] = {}
        body = text
        if text.startswith("---\n"):
            parts = text.split("---\n", 2)
            if len(parts) == 3:
                raw_metadata = parts[1]
                body = parts[2].strip()
                for line in raw_metadata.splitlines():
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
        relative_path = path.relative_to(self.knowledge_repo).as_posix()
        commit = self._file_commit(relative_path)
        return {
            "path": relative_path,
            "metadata": metadata,
            "body": body,
            "text": text,
            "commit": commit,
            "id": metadata.get("id") or slugify(relative_path),
            "team": metadata.get("team") or relative_path.split("/", 1)[0],
            "type": metadata.get("type", "note"),
            "entity": metadata.get("entity", ""),
            "scope": metadata.get("scope", ""),
            "title": metadata.get("title") or relative_path,
            "summary": metadata.get("summary") or body.splitlines()[0].lstrip("# ").strip(),
            "required_columns": parse_csv(metadata.get("required_columns")),
            "preferred_tables": parse_csv(metadata.get("preferred_tables")),
        }

    def _file_commit(self, relative_path: str) -> dict[str, Any] | None:
        result = self._git(
            ["log", "-1", "--format=%H%x1f%an%x1f%ae%x1f%aI%x1f%s", "--", relative_path],
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        commit_hash, author, email, authored_at, subject = result.stdout.strip().split("\x1f", 4)
        return {
            "hash": commit_hash,
            "short_hash": commit_hash[:8],
            "author": author,
            "email": email,
            "authored_at": authored_at,
            "subject": subject,
        }

    def all_documents(self) -> list[dict[str, Any]]:
        self.ensure_ready()
        docs = [
            self.parse_document(path)
            for path in sorted(self.knowledge_repo.rglob("*.md"))
            if ".git" not in path.parts
        ]
        return docs

    def visible_documents(self, user: User) -> list[dict[str, Any]]:
        return [doc for doc in self.all_documents() if user.can_read_team(doc["team"])]

    def get_document_by_id(self, user: User, document_id: str) -> dict[str, Any] | None:
        for doc in self.visible_documents(user):
            if doc["id"] == document_id:
                return doc
        return None

    def warehouse_schema(self, user: User | None = None) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._connect_warehouse() as db:
            tables = [row["name"] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
            schema = []
            for table in tables:
                if user and not user.can_access_table(table):
                    continue
                columns = [row["name"] for row in db.execute(f"PRAGMA table_info({self._quote_identifier(table)})")]
                schema.append({"table": table, "columns": columns})
            return schema

    def sample_values(self, table: str, columns: list[str]) -> dict[str, list[Any]]:
        with self._connect_warehouse() as db:
            available = set(self._columns_for_table(table))
            samples: dict[str, list[Any]] = {}
            for column in columns:
                if column not in available:
                    samples[column] = []
                    continue
                query = f"SELECT DISTINCT {self._quote_identifier(column)} AS value FROM {self._quote_identifier(table)} WHERE {self._quote_identifier(column)} IS NOT NULL LIMIT 5"
                samples[column] = [row["value"] for row in db.execute(query)]
            return samples

    def _columns_for_table(self, table: str) -> list[str]:
        with self._connect_warehouse() as db:
            return [row["name"] for row in db.execute(f"PRAGMA table_info({self._quote_identifier(table)})")]

    def _quote_identifier(self, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError(f"Unsafe identifier: {value}")
        return f'"{value}"'

    def _accessible_matching_tables(self, user: User, required_columns: list[str]) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for table in self.warehouse_schema(user):
            columns = set(table["columns"])
            missing = [column for column in required_columns if column not in columns]
            if not missing:
                matches.append({"table": table["table"], "columns": table["columns"]})
        return matches

    def prepare_calculation(self, question: str, user_id: str | None = None) -> dict[str, Any]:
        user = self.get_user(user_id)
        query_tokens = tokenize(question)
        wants_arr = "arr" in query_tokens or {"annual", "recurring", "revenue"} & query_tokens
        definitions = []
        for doc in self.visible_documents(user):
            if doc["type"] != "definition":
                continue
            if wants_arr and doc["entity"].lower() != "arr":
                continue
            matching_tables = self._accessible_matching_tables(user, doc["required_columns"])
            if not matching_tables:
                continue
            definitions.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "team": doc["team"],
                    "scope": doc["scope"],
                    "summary": doc["summary"],
                    "entails": self._definition_entails(doc),
                    "required_columns": doc["required_columns"],
                    "preferred_tables": doc["preferred_tables"],
                    "accessible_tables": matching_tables,
                    "citation": self._citation(doc),
                }
            )
        definitions.sort(key=lambda item: (item["team"], item["scope"]))
        return {
            "question": question,
            "user": self._user_payload(user),
            "entity": "ARR" if wants_arr else None,
            "requires_choice": len(definitions) > 1,
            "definitions": definitions,
            "message": "Multiple ARR definitions are visible. Choose the one that matches your business scope."
            if len(definitions) > 1
            else "One matching definition is visible."
            if definitions
            else "No accessible committed definition was found.",
        }

    def _definition_entails(self, doc: dict[str, Any]) -> list[str]:
        if doc["id"] == "arr-finance-board":
            return [
                "Use active subscriptions only.",
                "Sum monthly_recurring_revenue.",
                "Multiply by 12 for annualized board reporting.",
            ]
        if doc["id"] == "arr-renewals-forecast":
            return [
                "Use renewal opportunities already represented as annual values.",
                "Include committed and likely renewals.",
                "Do not multiply by 12 again.",
            ]
        return [doc["summary"]]

    def run_calculation(self, user_id: str | None, definition_id: str, table: str) -> dict[str, Any]:
        user = self.get_user(user_id)
        if not user.can_access_table(table):
            raise PermissionError(f"{user.name} cannot access table {table}")
        doc = self.get_document_by_id(user, definition_id)
        if not doc:
            raise PermissionError(f"Definition {definition_id} is not visible to {user.name}")
        columns = set(self._columns_for_table(table))
        missing = [column for column in doc["required_columns"] if column not in columns]
        if missing:
            return {
                "ok": False,
                "blocked": True,
                "message": "Selected table is missing required columns.",
                "missing_columns": missing,
                "definition": self._doc_option(doc),
                "table": table,
            }
        sql_template = doc["metadata"].get("formula_sql", "")
        sql = self._safe_sql_from_template(sql_template, table)
        with self._connect_warehouse() as db:
            row = db.execute(sql).fetchone()
        value = row[0] if row else None
        return {
            "ok": True,
            "definition": self._doc_option(doc),
            "table": table,
            "sql": sql,
            "result": {"label": "ARR", "value": value},
            "citation": self._citation(doc),
        }

    def _safe_sql_from_template(self, sql_template: str, table: str) -> str:
        if "{table}" not in sql_template:
            raise ValueError("Committed formula must contain {table}")
        quoted_table = self._quote_identifier(table)
        sql = sql_template.replace("{table}", quoted_table)
        normalized = re.sub(r"\s+", " ", sql.strip()).lower()
        forbidden = (";", "--", "/*", "*/", " insert ", " update ", " delete ", " drop ", " alter ", " attach ", " pragma ")
        if not normalized.startswith("select ") or any(token in f" {normalized} " for token in forbidden):
            raise ValueError("Only single read-only SELECT calculations are allowed")
        return sql

    def create_author_proposal(
        self,
        user_id: str | None,
        natural_language: str,
        target_team: str | None = None,
    ) -> dict[str, Any]:
        user = self.get_user(user_id)
        inferred = self._infer_proposal(natural_language, target_team)
        proposal_id = f"prop_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
        markdown = self._proposal_markdown(inferred, natural_language, user)
        with self._connect_app() as db:
            db.execute(
                """
                INSERT INTO proposals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal_id,
                    utc_now(),
                    user.id,
                    natural_language,
                    inferred["team"],
                    inferred["path"],
                    json.dumps(inferred, sort_keys=True),
                    markdown,
                    "draft",
                ),
            )
        validation = self.validate_proposal(proposal_id, user.id)
        return {
            "proposal_id": proposal_id,
            "proposal": inferred,
            "markdown": markdown,
            "validation": validation,
            "model": model_config()["reasoning"],
            "model_config": model_config(),
            "authoring_source": inferred.get("authoring_source", "local_deterministic"),
        }

    def _infer_proposal(self, natural_language: str, target_team: str | None) -> dict[str, Any]:
        config = model_config()
        if config["mode"] == "openai_configured":
            try:
                return self._infer_proposal_with_openai(natural_language, target_team, config)
            except Exception as error:
                if configured_env_value("DATAMETA_OPENAI_FALLBACK") == "deterministic":
                    fallback = self._infer_proposal_deterministic(natural_language, target_team)
                    fallback["authoring_source"] = "local_deterministic_after_openai_error"
                    fallback["authoring_error"] = str(error)
                    return fallback
                raise
        if config["mode"] == "openai_missing_model":
            raise ValueError("OPENAI_API_KEY is configured, but no authoring model is set. Add DATAMETA_REASONING_MODEL or OPENAI_MODEL to .env.")
        return self._infer_proposal_deterministic(natural_language, target_team)

    def _infer_proposal_with_openai(
        self,
        natural_language: str,
        target_team: str | None,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        user_context = {
            "target_team": target_team,
            "valid_teams": sorted({user_team for user in SEEDED_USERS.values() for user_team in user.read_teams + user.write_teams}),
            "warehouse_schema": self.warehouse_schema(),
            "existing_documents": [
                {
                    "id": doc["id"],
                    "path": doc["path"],
                    "type": doc["type"],
                    "entity": doc["entity"],
                    "scope": doc["scope"],
                    "team": doc["team"],
                    "title": doc["title"],
                    "summary": doc["summary"],
                    "required_columns": doc["required_columns"],
                    "preferred_tables": doc["preferred_tables"],
                }
                for doc in self.all_documents()
            ],
        }
        system = (
            "You convert analyst comments into DataMeta knowledge proposals. "
            "Return JSON that exactly matches the schema. Choose metadata that can become a markdown file, "
            "a searchable document, and Neo4j graph nodes/relationships. "
            "Use only teams and warehouse columns from the supplied context. "
            "For calculation definitions, write read-only SQLite SELECT formula_sql using {table}; leave formula_sql empty for non-calculation knowledge. "
            "Prefer updating the same entity/scope when a comment says change or redefine an existing concept. "
            "Do not invent inaccessible tables or columns."
        )
        user = json.dumps(
            {
                "comment": natural_language,
                "context": user_context,
            },
            indent=2,
            sort_keys=True,
        )
        proposal = self._authoring_client(config).structured_json(
            system=system,
            user=user,
            schema_name="datameta_knowledge_proposal",
            schema=PROPOSAL_SCHEMA,
        )
        proposal = self._normalize_model_proposal(proposal, natural_language, target_team)
        proposal["authoring_source"] = "openai_responses"
        return proposal

    def _authoring_client(self, config: dict[str, Any]) -> OpenAIResponsesClient:
        if self.openai_client:
            return self.openai_client
        api_key = configured_env_value("OPENAI_API_KEY")
        model = config.get("reasoning")
        if not api_key or not model:
            raise RuntimeError("OPENAI_API_KEY and DATAMETA_REASONING_MODEL are required for OpenAI authoring")
        return OpenAIResponsesClient(api_key=api_key, model=model)

    def _normalize_model_proposal(
        self,
        proposal: dict[str, Any],
        natural_language: str,
        target_team: str | None,
    ) -> dict[str, Any]:
        normalized = dict(proposal)
        if target_team:
            normalized["team"] = target_team
        normalized["team"] = slugify(str(normalized.get("team") or "analytics"))
        valid_teams = {team for user in SEEDED_USERS.values() for team in user.read_teams + user.write_teams}
        if normalized["team"] not in valid_teams:
            raise ValueError(f"OpenAI returned unknown team: {normalized['team']}")
        normalized["type"] = str(normalized.get("type") or "note")
        if normalized["type"] not in {"definition", "policy", "runbook", "note"}:
            normalized["type"] = "note"
        normalized["entity"] = str(normalized.get("entity") or "Operational Note").strip()[:80]
        normalized["scope"] = slugify(str(normalized.get("scope") or "general")).replace("-", "_")
        normalized["id"] = slugify(str(normalized.get("id") or f"{normalized['entity']}-{normalized['team']}-{normalized['scope']}"))
        normalized["title"] = str(normalized.get("title") or normalized["entity"]).strip()[:120]
        normalized["summary"] = str(normalized.get("summary") or natural_language.strip()).strip().rstrip(".")[:240] + "."
        normalized["formula_sql"] = str(normalized.get("formula_sql") or "").strip()
        normalized["required_columns"] = self._clean_string_list(normalized.get("required_columns"))
        normalized["preferred_tables"] = self._clean_string_list(normalized.get("preferred_tables"))
        normalized["search_terms"] = self._clean_string_list(normalized.get("search_terms"))
        normalized["neo4j_labels"] = self._clean_string_list(normalized.get("neo4j_labels"))
        normalized["neo4j_relationships"] = self._clean_relationships(normalized.get("neo4j_relationships"))
        normalized["body_markdown"] = str(normalized.get("body_markdown") or natural_language.strip()).strip()
        normalized["path"] = self._safe_proposal_path(normalized)
        if normalized["formula_sql"]:
            self._validate_formula_template(normalized["formula_sql"])
        return normalized

    def _clean_string_list(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        cleaned = []
        for value in values:
            item = str(value).strip()
            if item and item not in cleaned:
                cleaned.append(item)
        return cleaned

    def _clean_relationships(self, values: Any) -> list[dict[str, str]]:
        if not isinstance(values, list):
            return []
        relationships = []
        for value in values:
            if not isinstance(value, dict):
                continue
            relationship_type = slugify(str(value.get("type", ""))).replace("-", "_").upper()
            target = str(value.get("target", "")).strip()
            if relationship_type and target:
                relationships.append({"type": relationship_type, "target": target[:120]})
        return relationships

    def _safe_proposal_path(self, proposal: dict[str, Any]) -> str:
        path = str(proposal.get("path") or "").strip()
        if not path or ".." in Path(path).parts or path.startswith("/"):
            path = f"{proposal['team']}/{slugify(proposal['entity'])}-{slugify(proposal['scope'])}.md"
        if not path.endswith(".md"):
            path = f"{path}.md"
        parts = [slugify(part.removesuffix(".md")) for part in path.split("/") if part]
        if not parts:
            parts = [proposal["team"], proposal["id"]]
        if parts[0] != proposal["team"]:
            parts.insert(0, proposal["team"])
        return "/".join(parts[:-1] + [f"{parts[-1]}.md"])

    def _validate_formula_template(self, formula_sql: str) -> None:
        if "{table}" not in formula_sql:
            raise ValueError("OpenAI returned formula_sql without {table}")
        normalized = re.sub(r"\s+", " ", formula_sql.strip()).lower()
        forbidden = (";", "--", "/*", "*/", " insert ", " update ", " delete ", " drop ", " alter ", " attach ", " pragma ")
        if not normalized.startswith("select ") or any(token in f" {normalized} " for token in forbidden):
            raise ValueError("OpenAI returned unsafe formula_sql")

    def _infer_proposal_deterministic(self, natural_language: str, target_team: str | None) -> dict[str, Any]:
        text = natural_language.lower()
        team = target_team or (
            "renewals"
            if "renewal" in text
            else "finance"
            if "finance" in text or "arr" in text
            else "ops"
            if "outlier" in text or "ops" in text
            else "analytics"
        )
        entity = "ARR" if "arr" in text or "recurring revenue" in text else "Operational Note"
        scope = "board_reporting" if "board" in text or team == "finance" else "renewal_forecasting" if team == "renewals" else "general"
        title = f"{entity} for {scope.replace('_', ' ')}"
        if entity == "ARR" and team == "renewals":
            formula_sql = "SELECT ROUND(SUM(renewal_arr), 2) AS arr FROM {table} WHERE renewal_status IN ('committed', 'likely')"
            required_columns = ["renewal_arr", "renewal_status"]
            preferred_tables = ["renewals"]
        elif entity == "ARR":
            formula_sql = "SELECT ROUND(SUM(monthly_recurring_revenue) * 12, 2) AS arr FROM {table} WHERE status = 'active'"
            required_columns = ["monthly_recurring_revenue", "status"]
            preferred_tables = ["subscriptions"]
        else:
            formula_sql = ""
            required_columns = []
            preferred_tables = []
        path = f"{team}/{slugify(entity)}-{slugify(scope)}.md"
        return {
            "id": f"{slugify(entity)}-{team}-{slugify(scope)}",
            "type": "definition" if entity == "ARR" else "note",
            "entity": entity,
            "scope": scope,
            "team": team,
            "title": title,
            "summary": natural_language.strip().rstrip(".") + ".",
            "formula_sql": formula_sql,
            "required_columns": required_columns,
            "preferred_tables": preferred_tables,
            "path": path,
            "body_markdown": natural_language.strip(),
            "search_terms": sorted(tokenize(natural_language))[:12],
            "neo4j_labels": ["Document", "Definition"] if entity == "ARR" else ["Document", "Note"],
            "neo4j_relationships": [{"type": "OWNED_BY", "target": team}],
            "authoring_source": "local_deterministic",
        }

    def _proposal_markdown(self, proposal: dict[str, Any], natural_language: str, user: User) -> str:
        required = ",".join(proposal["required_columns"])
        preferred = ",".join(proposal["preferred_tables"])
        search_terms = ",".join(proposal.get("search_terms", []))
        neo4j_labels = ",".join(proposal.get("neo4j_labels", []))
        neo4j_relationships = json.dumps(proposal.get("neo4j_relationships", []), sort_keys=True)
        body = str(proposal.get("body_markdown") or natural_language.strip()).strip()
        return f"""---
id: {proposal["id"]}
type: {proposal["type"]}
entity: {proposal["entity"]}
scope: {proposal["scope"]}
team: {proposal["team"]}
title: {proposal["title"]}
summary: {proposal["summary"]}
formula_sql: {proposal["formula_sql"]}
required_columns: {required}
preferred_tables: {preferred}
search_terms: {search_terms}
neo4j_labels: {neo4j_labels}
neo4j_relationships: {neo4j_relationships}
authoring_source: {proposal.get("authoring_source", "local_deterministic")}
updated_by: {user.name}
updated_at: {utc_now()}
---
# {proposal["title"]}

{body}

Captured by DataMeta from natural language and awaiting analyst confirmation.
"""

    def validate_proposal(self, proposal_id: str, user_id: str | None = None) -> dict[str, Any]:
        proposal = self._load_proposal(proposal_id)
        user = self.get_user(user_id or proposal["created_by"])
        metadata = proposal["metadata"]
        checks: list[dict[str, Any]] = []
        can_write = user.can_write_team(metadata["team"])
        checks.append(
            {
                "name": "RBAC write access",
                "ok": can_write,
                "detail": f"{user.name} {'can' if can_write else 'cannot'} write to {metadata['team']}.",
            }
        )
        matching_tables = self._accessible_matching_tables(user, metadata.get("required_columns", []))
        if metadata.get("required_columns"):
            checks.append(
                {
                    "name": "Schema and sample validation",
                    "ok": bool(matching_tables),
                    "detail": "At least one accessible table has the required fields."
                    if matching_tables
                    else "No accessible table has every required field.",
                    "matching_tables": matching_tables,
                }
            )
        conflicts = self._detect_conflicts(metadata)
        checks.append(
            {
                "name": "Same entity + scope conflict check",
                "ok": not conflicts,
                "detail": "No conflicting committed knowledge found."
                if not conflicts
                else "Committed knowledge with the same entity and scope already exists.",
                "conflicts": conflicts,
            }
        )
        samples = {}
        for table in matching_tables:
            samples[table["table"]] = self.sample_values(table["table"], metadata.get("required_columns", []))
        blocked = not can_write or (metadata.get("required_columns") and not matching_tables)
        needs_confirmation = bool(conflicts)
        return {
            "proposal_id": proposal_id,
            "ok": not blocked,
            "blocked": blocked,
            "needs_confirmation": needs_confirmation,
            "policy": {
                "conflict_mode": "confirm_before_overwrite",
                "rbac": "team_folder_and_table_least_privilege",
            },
            "checks": checks,
            "samples": samples,
        }

    def _load_proposal(self, proposal_id: str) -> dict[str, Any]:
        with self._connect_app() as db:
            row = db.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
        if not row:
            raise ValueError(f"Unknown proposal: {proposal_id}")
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "created_by": row["created_by"],
            "natural_language": row["natural_language"],
            "target_team": row["target_team"],
            "target_path": row["target_path"],
            "metadata": json.loads(row["metadata_json"]),
            "markdown": row["markdown"],
            "status": row["status"],
        }

    def _detect_conflicts(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        conflicts = []
        for doc in self.all_documents():
            if doc["entity"].lower() == metadata.get("entity", "").lower() and doc["scope"] == metadata.get("scope"):
                conflicts.append(
                    {
                        "id": doc["id"],
                        "path": doc["path"],
                        "team": doc["team"],
                        "title": doc["title"],
                        "summary": doc["summary"],
                        "citation": self._citation(doc),
                    }
                )
        return conflicts

    def commit_proposal(
        self,
        user_id: str | None,
        proposal_id: str,
        confirm_overwrite: bool = False,
    ) -> dict[str, Any]:
        proposal = self._load_proposal(proposal_id)
        user = self.get_user(user_id or proposal["created_by"])
        validation = self.validate_proposal(proposal_id, user.id)
        if validation["blocked"]:
            raise PermissionError("Proposal is blocked by validation or RBAC")
        if validation["needs_confirmation"] and not confirm_overwrite:
            return {
                "ok": False,
                "needs_confirmation": True,
                "message": "Existing same-entity/same-scope knowledge found. Confirm overwrite to commit.",
                "validation": validation,
                "confirmed_by": None,
            }
        path = self.knowledge_repo / proposal["target_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        markdown = proposal["markdown"]
        if validation["needs_confirmation"]:
            markdown += (
                f"\n\n## Overwrite confirmation\n\nConfirmed by {user.name} on {utc_now()}.\n"
                "This change was committed after DataMeta surfaced same-entity/same-scope knowledge.\n"
            )
        path.write_text(markdown, encoding="utf-8")
        self._git(["add", proposal["target_path"]])
        commit_hash = self._git_commit(
            f"DataMeta: update {proposal['metadata']['entity']} for {proposal['metadata']['scope']}",
            user.name,
            f"{user.id}@shoppy.local",
        )
        doc = self.parse_document(path)
        neo4j_result = self.sync_document_to_neo4j(doc)
        with self._connect_app() as db:
            db.execute("UPDATE proposals SET status = ? WHERE id = ?", ("committed", proposal_id))
        return {
            "ok": True,
            "proposal_id": proposal_id,
            "path": proposal["target_path"],
            "commit_hash": commit_hash,
            "confirmed_by": user.name,
            "confirmed_at": utc_now(),
            "validation": validation,
            "neo4j": neo4j_result,
        }

    def sync_document_to_neo4j(self, doc: dict[str, Any]) -> dict[str, Any]:
        url = configured_env_value("DATAMETA_NEO4J_URL")
        user = configured_env_value("DATAMETA_NEO4J_USER")
        password = configured_env_value("DATAMETA_NEO4J_PASSWORD")
        if not url or not user or not password:
            return {"ok": False, "status": "not_configured"}
        labels = ["Document", *parse_csv(doc["metadata"].get("neo4j_labels"))]
        if doc["type"] == "definition":
            labels.append("Definition")
        if doc["type"] == "runbook":
            labels.append("Runbook")
        if doc["type"] == "policy":
            labels.append("Policy")
        labels = [label for label in dict.fromkeys(self._safe_neo4j_label(label) for label in labels) if label]
        label_clause = "".join(f":{label}" for label in labels)
        statements = [
            {
                "statement": (
                    f"MERGE (d{label_clause} {{id: $id}}) "
                    "SET d.path = $path, d.type = $type, d.entity = $entity, d.scope = $scope, "
                    "d.team = $team, d.title = $title, d.summary = $summary, d.body = $body, "
                    "d.search_terms = $search_terms, d.updated_at = $updated_at "
                    "WITH d "
                    "MERGE (team:Team {name: $team}) "
                    "MERGE (d)-[:OWNED_BY]->(team) "
                    "WITH d "
                    "MERGE (entity:Entity {name: $entity}) "
                    "MERGE (d)-[:ABOUT]->(entity)"
                ),
                "parameters": {
                    "id": doc["id"],
                    "path": doc["path"],
                    "type": doc["type"],
                    "entity": doc["entity"],
                    "scope": doc["scope"],
                    "team": doc["team"],
                    "title": doc["title"],
                    "summary": doc["summary"],
                    "body": doc["body"],
                    "search_terms": parse_csv(doc["metadata"].get("search_terms")),
                    "updated_at": doc["metadata"].get("updated_at"),
                },
            }
        ]
        for relationship in self._relationships_from_metadata(doc["metadata"].get("neo4j_relationships")):
            rel_type = self._safe_neo4j_label(relationship["type"])
            if not rel_type:
                continue
            statements.append(
                {
                    "statement": (
                        "MATCH (d:Document {id: $id}) "
                        "MERGE (target:Entity {name: $target}) "
                        f"MERGE (d)-[:{rel_type}]->(target)"
                    ),
                    "parameters": {"id": doc["id"], "target": relationship["target"]},
                }
            )
        request_payload = {"statements": statements}
        token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        endpoint = f"{url.rstrip('/')}/db/neo4j/tx/commit"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            return {"ok": False, "status": "sync_failed", "error": str(error)}
        errors = payload.get("errors") or []
        return {"ok": not errors, "status": "synced" if not errors else "sync_failed", "errors": errors}

    def _safe_neo4j_label(self, value: str) -> str:
        candidate = re.sub(r"[^A-Za-z0-9_]", "_", str(value).strip())
        if not candidate or candidate[0].isdigit():
            return ""
        return candidate

    def _relationships_from_metadata(self, value: str | None) -> list[dict[str, str]]:
        if not value:
            return []
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            return []
        return self._clean_relationships(raw)

    def retrieve(self, user_id: str | None, query: str, include_trace: bool = False) -> dict[str, Any]:
        user = self.get_user(user_id)
        query_tokens = tokenize(query)
        if "recurring" in query_tokens or "revenue" in query_tokens:
            query_tokens.add("arr")
        scored = []
        for doc in self.visible_documents(user):
            haystack = " ".join(
                [
                    doc["title"],
                    doc["summary"],
                    doc["entity"],
                    doc["scope"],
                    doc["team"],
                    doc["body"],
                ]
            )
            doc_tokens = tokenize(haystack)
            score = len(query_tokens & doc_tokens)
            if "arr" in query_tokens and doc["entity"].lower() == "arr":
                score += 5
            if "outlier" in query_tokens and "outlier" in doc["title"].lower():
                score += 4
            if {"pipeline", "runbook", "ranking", "chart"} & query_tokens and doc["type"] == "runbook":
                score += 4
            if score:
                scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        packets = [
            {
                "score": score,
                "id": doc["id"],
                "path": doc["path"],
                "team": doc["team"],
                "type": doc["type"],
                "entity": doc["entity"],
                "scope": doc["scope"],
                "title": doc["title"],
                "summary": doc["summary"],
                "snippet": doc["body"][:900],
                "citation": self._citation(doc),
            }
            for score, doc in scored[:8]
        ]
        trace = None
        if include_trace:
            trace = {
                "retrieval_mode": "hybrid_keyword_semantic_graph_ready",
                "visible_teams": list(user.read_teams) if not user.is_admin else ["*"],
                "candidate_documents": len(self.visible_documents(user)),
                "returned_documents": len(packets),
                "graph_hops": [
                    "Question -> Definition/Runbook candidate",
                    "Definition -> required columns",
                    "Definition -> source table",
                    "Document -> commit metadata",
                ],
                "neo4j_status": "not_connected_local_fallback_active",
            }
        return {"query": query, "user": self._user_payload(user), "packets": packets, "trace": trace}

    def answer(self, user_id: str | None, question: str, include_trace: bool = False) -> dict[str, Any]:
        retrieval = self.retrieve(user_id, question, include_trace)
        arr_flow = self.prepare_calculation(question, user_id) if "arr" in question.lower() else None
        if arr_flow and arr_flow["definitions"]:
            answer_text = (
                "DataMeta found multiple latest ARR definitions visible to you. "
                "Finance is for board reporting and multiplies active subscription MRR by 12. "
                "Renewals is for renewal forecasting and sums already annualized committed/likely renewal ARR. "
                "Choose a definition before running the calculation."
            )
        elif retrieval["packets"]:
            names = ", ".join(packet["title"] for packet in retrieval["packets"][:3])
            answer_text = f"DataMeta found grounded Shoppy knowledge for: {names}."
        else:
            answer_text = "DataMeta could not find accessible committed Shoppy knowledge for this question."
        return {
            "question": question,
            "answer": answer_text,
            "calculation_prompt": arr_flow,
            "citations": [packet["citation"] for packet in retrieval["packets"]],
            "trace": retrieval["trace"],
        }

    def ask(self, user_id: str | None, question: str, include_trace: bool = False) -> dict[str, Any]:
        tokens = tokenize(question)
        data_quality_tokens = {
            "anomaly",
            "bad",
            "incorrect",
            "issue",
            "outlier",
            "quality",
            "spike",
            "strange",
            "suspicious",
            "wrong",
        }
        if tokens & data_quality_tokens:
            return self._triage_data_quality_question(user_id, question, include_trace)
        answer = self.answer(user_id, question, include_trace)
        return {"intent": "answer", "action": "datameta_answer", **answer}

    def _triage_data_quality_question(self, user_id: str | None, question: str, include_trace: bool) -> dict[str, Any]:
        user = self.get_user(user_id)
        table_name = self._infer_table_from_text(question, user)
        subject = self._infer_outlier_subject(question, table_name)
        description = question.strip()
        missing = []
        if not table_name:
            missing.append("table_name")
        if not subject:
            missing.append("subject")
        if not self._has_specific_outlier_detail(question, table_name):
            missing.append("description")
        retrieval = self.retrieve(user.id, question, include_trace)
        if missing:
            return {
                "intent": "flag_outlier",
                "action": "needs_more_detail",
                "ok": False,
                "needs_more_detail": True,
                "message": (
                    "I can help flag this for data-owner review. "
                    "Please provide the table or metric, what looks suspicious, and a short description."
                ),
                "missing": missing,
                "available_tables": list(user.tables),
                "example": "orders table, ord_005 May 4 GMV spike, GMV is more than 10x the prior three days.",
                "citations": [packet["citation"] for packet in retrieval["packets"]],
                "trace": retrieval["trace"],
            }
        flag = self.flag_outlier(user.id, table_name, subject, description)
        return {
            "intent": "flag_outlier",
            "action": "datameta_flag_outlier",
            "ok": True,
            "message": "DataMeta flagged the suspicious data point for owner-team review.",
            "flag": flag,
            "citations": [packet["citation"] for packet in retrieval["packets"]],
            "trace": retrieval["trace"],
        }

    def _infer_table_from_text(self, text: str, user: User) -> str | None:
        normalized = text.lower()
        for table in user.tables:
            table_words = table.replace("_", " ")
            if table in normalized or table_words in normalized:
                return table
        if {"order", "orders", "gmv", "refund", "marketplace"} & tokenize(text) and user.can_access_table("orders"):
            return "orders"
        if {"metric", "metrics", "daily"} & tokenize(text) and user.can_access_table("ops_daily_metrics"):
            return "ops_daily_metrics"
        if {"renewal", "renewals"} & tokenize(text) and user.can_access_table("renewals"):
            return "renewals"
        if {"subscription", "subscriptions", "mrr"} & tokenize(text) and user.can_access_table("subscriptions"):
            return "subscriptions"
        return None

    def _infer_outlier_subject(self, text: str, table_name: str | None) -> str:
        cleaned = re.sub(r"\s+", " ", text.strip()).rstrip(".")
        if not cleaned:
            return ""
        vague_tokens = {"data", "datapoint", "point", "seems", "suspicious", "looks", "bad", "wrong"}
        if tokenize(cleaned) <= vague_tokens:
            return ""
        if table_name and table_name.replace("_", " ") not in cleaned.lower():
            return f"{table_name} issue: {cleaned[:70]}"
        return cleaned[:90]

    def _has_specific_outlier_detail(self, text: str, table_name: str | None) -> bool:
        tokens = tokenize(text)
        vague = tokens <= {"data", "datapoint", "point", "seems", "suspicious", "looks", "bad", "wrong"}
        if vague:
            return False
        return bool(table_name and (len(tokens) >= 5 or re.search(r"\d", text)))

    def flag_outlier(
        self,
        user_id: str | None,
        table_name: str,
        subject: str,
        description: str,
        owner_team: str = "data-ownership",
    ) -> dict[str, Any]:
        user = self.get_user(user_id)
        if not user.can_access_table(table_name):
            raise PermissionError(f"{user.name} cannot access table {table_name}")
        flag_id = f"flag_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
        with self._connect_app() as db:
            db.execute(
                "INSERT INTO outlier_flags VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    flag_id,
                    utc_now(),
                    user.id,
                    owner_team,
                    table_name,
                    subject,
                    description,
                    "pending_owner_review",
                    None,
                    None,
                    None,
                ),
            )
        return {
            "ok": True,
            "flag_id": flag_id,
            "status": "pending_owner_review",
            "owner_team": owner_team,
            "created_by": self._user_payload(user),
        }

    def resolve_flag(self, user_id: str | None, flag_id: str, resolution: str) -> dict[str, Any]:
        user = self.get_user(user_id)
        with self._connect_app() as db:
            row = db.execute("SELECT * FROM outlier_flags WHERE id = ?", (flag_id,)).fetchone()
            if not row:
                raise ValueError(f"Unknown flag: {flag_id}")
            if not user.can_write_team(row["owner_team"]):
                raise PermissionError(f"{user.name} cannot resolve flags owned by {row['owner_team']}")
            resolved_at = utc_now()
            db.execute(
                "UPDATE outlier_flags SET status = ?, resolved_at = ?, resolved_by = ?, resolution = ? WHERE id = ?",
                ("resolved", resolved_at, user.id, resolution, flag_id),
            )
        relative_path = f"data-ownership/resolved-flags/{flag_id}.md"
        path = self.knowledge_repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"""---
id: {flag_id}
type: data_quality_flag
entity: Data Quality Flag
scope: {row["table_name"]}
team: data-ownership
title: {row["subject"]}
summary: {resolution}
updated_by: {user.name}
updated_at: {resolved_at}
---
# {row["subject"]}

Flagged by: {row["created_by"]}

Table: `{row["table_name"]}`

Issue:

{row["description"]}

Resolution:

{resolution}
""",
            encoding="utf-8",
        )
        self._git(["add", relative_path])
        commit_hash = self._git_commit(f"DataMeta: resolve outlier {flag_id}", user.name, f"{user.id}@shoppy.local")
        return {
            "ok": True,
            "flag_id": flag_id,
            "status": "resolved",
            "resolved_by": self._user_payload(user),
            "resolved_at": resolved_at,
            "commit_hash": commit_hash,
            "path": relative_path,
        }

    def list_flags(self) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._connect_app() as db:
            rows = db.execute("SELECT * FROM outlier_flags ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def run_pipeline(self, user_id: str | None, runbook_id: str = "gmv-category-ranker", variant: str | None = None) -> dict[str, Any]:
        user = self.get_user(user_id)
        doc = self.get_document_by_id(user, runbook_id)
        if not doc or doc["type"] != "runbook":
            raise PermissionError(f"Runbook {runbook_id} is not visible to {user.name}")
        if not user.can_access_table("orders"):
            raise PermissionError(f"{user.name} cannot access orders")
        sql = """
            SELECT
                category,
                ROUND(SUM(gross_merchandise_value - refund_amount), 2) AS net_gmv,
                COUNT(*) AS order_count
            FROM orders
            GROUP BY category
            ORDER BY net_gmv DESC
            LIMIT 5
        """
        with self._connect_warehouse() as db:
            rows = [dict(row) for row in db.execute(sql)]
        colors = parse_csv(doc["metadata"].get("chart_colors"))
        chart = [
            {
                "label": row["category"],
                "value": row["net_gmv"],
                "color": colors[index % len(colors)] if colors else "#2563eb",
            }
            for index, row in enumerate(rows)
        ]
        output = {
            "runbook": self._doc_option(doc),
            "variant": variant or "weekly_ops_review",
            "sql": re.sub(r"\s+", " ", sql).strip(),
            "table": rows,
            "chart": chart,
            "citation": self._citation(doc),
        }
        run_id = f"run_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
        with self._connect_app() as db:
            db.execute(
                "INSERT INTO pipeline_runs VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, utc_now(), user.id, runbook_id, variant, json.dumps(output, sort_keys=True)),
            )
        return {"ok": True, "run_id": run_id, "output": output}

    def list_pipeline_runs(self) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._connect_app() as db:
            rows = db.execute("SELECT * FROM pipeline_runs ORDER BY created_at DESC").fetchall()
        runs = []
        for row in rows:
            item = dict(row)
            item["output"] = json.loads(item.pop("output_json"))
            runs.append(item)
        return runs

    def history(self, path: str | None = None) -> dict[str, Any]:
        self.ensure_ready()
        args = ["log", "--pretty=%H%x1f%an%x1f%ae%x1f%aI%x1f%s", "--name-only"]
        if path:
            args.extend(["--", path])
        result = self._git(args, check=False)
        commits = []
        current: dict[str, Any] | None = None
        for line in result.stdout.splitlines():
            if "\x1f" in line:
                if current:
                    commits.append(current)
                commit_hash, author, email, authored_at, subject = line.split("\x1f", 4)
                current = {
                    "hash": commit_hash,
                    "short_hash": commit_hash[:8],
                    "author": author,
                    "email": email,
                    "authored_at": authored_at,
                    "subject": subject,
                    "paths": [],
                }
            elif line.strip() and current:
                current["paths"].append(line.strip())
        if current:
            commits.append(current)
        return {
            "commits": commits,
            "documents": [
                {
                    "id": doc["id"],
                    "path": doc["path"],
                    "team": doc["team"],
                    "type": doc["type"],
                    "entity": doc["entity"],
                    "scope": doc["scope"],
                    "title": doc["title"],
                    "summary": doc["summary"],
                    "citation": self._citation(doc),
                }
                for doc in self.all_documents()
            ],
            "flags": self.list_flags(),
            "pipeline_runs": self.list_pipeline_runs(),
        }

    def bootstrap(self, user_id: str | None = None) -> dict[str, Any]:
        user = self.get_user(user_id)
        return {
            "project": "DataMeta",
            "company": "Shoppy",
            "models": model_config(),
            "user": self._user_payload(user),
            "users": self.users(),
            "schema": self.warehouse_schema(user),
            "history": self.history(),
            "mcp": {
                "url": "http://127.0.0.1:8000/mcp",
                "server_name": "datameta",
                "tools": self.mcp_tool_names(),
            },
        }

    def mcp_tool_names(self) -> list[str]:
        return [
            "datameta_ask",
            "datameta_retrieve",
            "datameta_answer",
            "datameta_author_proposal",
            "datameta_validate_proposal",
            "datameta_commit_proposal",
            "datameta_flag_outlier",
            "datameta_resolve_flag",
            "datameta_run_pipeline",
            "datameta_prepare_calculation",
            "datameta_run_calculation",
            "datameta_history",
        ]

    def _user_payload(self, user: User) -> dict[str, Any]:
        return {
            "id": user.id,
            "name": user.name,
            "roles": list(user.roles),
            "read_teams": list(user.read_teams),
            "write_teams": list(user.write_teams),
            "tables": list(user.tables),
        }

    def _citation(self, doc: dict[str, Any]) -> dict[str, Any]:
        commit = doc.get("commit") or {}
        return {
            "path": doc["path"],
            "commit_hash": commit.get("hash"),
            "commit": commit.get("short_hash"),
            "author": commit.get("author"),
            "authored_at": commit.get("authored_at"),
            "title": doc["title"],
        }

    def _doc_option(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": doc["id"],
            "title": doc["title"],
            "team": doc["team"],
            "scope": doc["scope"],
            "summary": doc["summary"],
            "required_columns": doc["required_columns"],
            "citation": self._citation(doc),
        }


def command_preview(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)
