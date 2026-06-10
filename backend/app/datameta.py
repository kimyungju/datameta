from __future__ import annotations

import json
import os
import re
import shlex
import sqlite3
import subprocess
import hashlib
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
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
DEFAULT_REASONING_MODEL = "gpt-5.5"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
LOCAL_EMBEDDING_DIMENSIONS = 256


def configured_env_value(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def neo4j_database() -> str:
    # Aura instances name the default database after the instance id (e.g. d1e35206),
    # not "neo4j". Allow overriding via env, falling back to the classic default.
    return configured_env_value("DATAMETA_NEO4J_DATABASE") or "neo4j"


def model_config() -> dict[str, Any]:
    api_key_configured = configured_env_value("OPENAI_API_KEY") is not None
    reasoning = configured_env_value("DATAMETA_REASONING_MODEL") or configured_env_value("OPENAI_MODEL") or DEFAULT_REASONING_MODEL
    embedding = configured_env_value("DATAMETA_EMBEDDING_MODEL") or configured_env_value("OPENAI_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
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


FOLDER_AGENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answerable", "findings"],
    "properties": {
        "answerable": {"type": "boolean"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["file_path", "heading", "summary", "snippet"],
                "properties": {
                    "file_path": {"type": "string"},
                    "heading": {"type": "string"},
                    "summary": {"type": "string"},
                    "snippet": {"type": "string"},
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

    def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        payload = {"model": model, "input": texts}
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
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
            raise RuntimeError(f"OpenAI Embeddings API failed: {error.code} {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"OpenAI Embeddings API could not be reached: {error.reason}") from error
        data = sorted(response_payload.get("data", []), key=lambda item: item.get("index", 0))
        embeddings = [item.get("embedding") for item in data]
        if len(embeddings) != len(texts) or any(not isinstance(item, list) for item in embeddings):
            raise RuntimeError("OpenAI Embeddings API returned an unexpected payload")
        return embeddings  # type: ignore[return-value]

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
        name="Jamie Incident Analyst",
        roles=("incident_analyst",),
        read_teams=(
            "security-incident-response",
            "legal-contracts",
            "customer-success-ops",
            "platform-operations",
            "vendor-risk-management",
            "data-governance",
        ),
        write_teams=(),
        tables=("sla_availability", "incident_events", "vendor_events"),
    ),
    "leah.legal": User(
        id="leah.legal",
        name="Leah Legal Counsel",
        roles=("legal", "author"),
        read_teams=("legal-contracts", "customer-success-ops", "platform-operations", "vendor-risk-management"),
        write_teams=("legal-contracts",),
        tables=("sla_availability", "incident_events"),
    ),
    "cam.cs": User(
        id="cam.cs",
        name="Cam Customer Success",
        roles=("customer_success", "author"),
        read_teams=("customer-success-ops", "legal-contracts", "platform-operations"),
        write_teams=("customer-success-ops",),
        tables=("sla_availability", "incident_events"),
    ),
    "peter.platform": User(
        id="peter.platform",
        name="Peter Platform Ops",
        roles=("platform_ops", "author"),
        read_teams=("platform-operations", "security-incident-response", "vendor-risk-management", "data-governance"),
        write_teams=("platform-operations",),
        tables=("sla_availability", "incident_events", "vendor_events"),
    ),
    "vera.vendor": User(
        id="vera.vendor",
        name="Vera Vendor Risk",
        roles=("vendor_risk", "author"),
        read_teams=("vendor-risk-management", "legal-contracts", "platform-operations"),
        write_teams=("vendor-risk-management",),
        tables=("vendor_events", "incident_events"),
    ),
    "sasha.security": User(
        id="sasha.security",
        name="Sasha Security IR",
        roles=("security_incident_response", "author"),
        read_teams=("security-incident-response", "platform-operations", "vendor-risk-management", "data-governance"),
        write_teams=("security-incident-response",),
        tables=("incident_events", "vendor_events"),
    ),
    "dina.data": User(
        id="dina.data",
        name="Dina Data Governance",
        roles=("data_governance", "author"),
        read_teams=("data-governance", "platform-operations", "legal-contracts", "vendor-risk-management"),
        write_teams=("data-governance",),
        tables=("sla_availability", "incident_events", "vendor_events"),
    ),
    "ops.associate": User(
        id="ops.associate",
        name="Olivia Operations Associate",
        roles=("operations", "analyst"),
        read_teams=("finance", "renewals", "sales"),
        write_teams=(),
        tables=("arr_subscriptions",),
    ),
    "ada.admin": User(
        id="ada.admin",
        name="Ada Admin",
        roles=("admin",),
        read_teams=(
            "security-incident-response",
            "legal-contracts",
            "customer-success-ops",
            "platform-operations",
            "vendor-risk-management",
            "data-governance",
            "finance",
            "renewals",
            "sales",
        ),
        write_teams=(
            "security-incident-response",
            "legal-contracts",
            "customer-success-ops",
            "platform-operations",
            "vendor-risk-management",
            "data-governance",
            "finance",
            "renewals",
            "sales",
        ),
        tables=("sla_availability", "incident_events", "vendor_events", "arr_subscriptions"),
    ),
}


class DataMetaService:
    def __init__(self, runtime_dir: Path | None = None, openai_client: OpenAIResponsesClient | None = None) -> None:
        root = PROJECT_ROOT
        configured = os.environ.get("DATAMETA_RUNTIME_DIR")
        default_runtime = Path("/tmp/datameta-runtime") if os.environ.get("VERCEL") else root / "runtime"
        self.runtime_dir = Path(configured) if configured else runtime_dir or default_runtime
        self.knowledge_repo = self.runtime_dir / "enterprise-incident-knowledge"
        self.app_db = self.runtime_dir / "datameta.sqlite"
        self.warehouse_db = self.runtime_dir / "enterprise-warehouse.sqlite"
        self.openai_client = openai_client
        self._multirepo_index: dict[str, Any] | None = None
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._ready = False
        self._neo4j_driver = None
        self._neo4j_driver_key: tuple[str, str, str] | None = None

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
                CREATE TABLE sla_availability (
                    customer_id TEXT NOT NULL,
                    month TEXT NOT NULL,
                    measured_availability REAL NOT NULL,
                    committed_availability REAL NOT NULL,
                    scheduled_maintenance_minutes REAL NOT NULL,
                    incident_id TEXT,
                    owner_team TEXT NOT NULL
                );

                CREATE TABLE incident_events (
                    event_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    customer_id TEXT,
                    service TEXT NOT NULL,
                    impact TEXT NOT NULL,
                    owner_team TEXT NOT NULL
                );

                CREATE TABLE vendor_events (
                    event_id TEXT PRIMARY KEY,
                    vendor_id TEXT NOT NULL,
                    incident_id TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    owner_team TEXT NOT NULL
                );

                CREATE TABLE arr_subscriptions (
                    subscription_id TEXT PRIMARY KEY,
                    customer TEXT NOT NULL,
                    region TEXT NOT NULL,
                    status TEXT NOT NULL,
                    renewal_status TEXT NOT NULL,
                    monthly_recurring_revenue REAL NOT NULL,
                    annual_contract_value REAL NOT NULL
                );
                """
            )
            db.executemany(
                "INSERT INTO arr_subscriptions VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    # ASEAN book of business. The three ARR definitions disagree on
                    # which rows/columns to count, so each returns a different number.
                    ("sub-001", "Acme Singapore", "ASEAN", "active", "committed", 20000.0, 240000.0),
                    ("sub-002", "Globex Malaysia", "ASEAN", "active", "committed", 15000.0, 180000.0),
                    ("sub-003", "Initech Thailand", "ASEAN", "active", "likely", 10000.0, 130000.0),
                    ("sub-004", "Umbrella Indonesia", "ASEAN", "churned", "lost", 8000.0, 0.0),
                    ("sub-005", "Hooli Philippines", "ASEAN", "active", "pipeline", 0.0, 90000.0),
                    ("sub-006", "Stark Vietnam", "ASEAN", "active", "committed", 12000.0, 150000.0),
                    # Non-ASEAN rows, excluded by every ASEAN definition.
                    ("sub-007", "Wayne EMEA", "EMEA", "active", "committed", 30000.0, 360000.0),
                    ("sub-008", "Cyberdyne Americas", "AMER", "active", "committed", 25000.0, 300000.0),
                ],
            )
            db.executemany(
                "INSERT INTO sla_availability VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    ("Customer A", "2026-05", 99.62, 99.90, 15.0, "inc-vendor-x-2026-05-20", "platform-operations"),
                    ("Customer B", "2026-05", 99.96, 99.50, 10.0, "inc-vendor-x-2026-05-20", "platform-operations"),
                    ("Customer C", "2026-05", 99.91, 99.90, 0.0, None, "platform-operations"),
                ],
            )
            db.executemany(
                "INSERT INTO incident_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    ("evt-001", "inc-vendor-x-2026-05-20", "2026-05-20T09:12:00+00:00", "Customer A", "public-api", "Elevated 5xx errors begin", "platform-operations"),
                    ("evt-002", "inc-vendor-x-2026-05-20", "2026-05-20T09:25:00+00:00", None, "session-service", "Vendor X confirmed as the source", "platform-operations"),
                    ("evt-003", "inc-vendor-x-2026-05-20", "2026-05-20T12:20:00+00:00", "Customer A", "public-api", "Customer A traffic restored", "platform-operations"),
                    ("evt-004", "inc-vendor-y-2026-04-11", "2026-04-11T03:10:00+00:00", "Customer B", "billing-export", "Vendor Y export delay", "platform-operations"),
                ],
            )
            db.executemany(
                "INSERT INTO vendor_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    ("vend-001", "Vendor X", "inc-vendor-x-2026-05-20", "2026-05-20T09:25:00+00:00", "sev-2", "Vendor X regional failover failed to trigger, degrading token validation.", "vendor-risk-management"),
                    ("vend-002", "Vendor X", "inc-vendor-x-2026-05-20", "2026-05-20T13:30:00+00:00", "sev-2", "Vendor X delivered preliminary restoration note.", "vendor-risk-management"),
                    ("vend-003", "Vendor Y", "inc-vendor-y-2026-04-11", "2026-04-11T04:00:00+00:00", "sev-3", "Vendor Y delayed non-production export.", "vendor-risk-management"),
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
            self._git_commit("Seed Generic Enterprise incident knowledge base", "DataMeta Seed", "seed@datameta.local")

    def _seed_markdown_files(self) -> dict[str, str]:
        files: dict[str, str] = {}

        repos = {
            "security-incident-response": {
                "team": "security-incident-response",
                "title": "Security Incident Response",
                "summary": "Incident classification, commander duties, breach assessment, and customer-safe response procedures for availability incidents.",
                "tags": "incident,response,security,classification,communications,vendor outage",
                "customers": "Customer A,Customer B,Customer C",
                "vendors": "Vendor X,Vendor Y",
            },
            "legal-contracts": {
                "team": "legal-contracts",
                "title": "Legal Contracts",
                "summary": "Customer agreement terms, availability SLA obligations, service credit rules, and notice requirements.",
                "tags": "legal,contracts,sla,availability,service credits,customer notices",
                "customers": "Customer A,Customer B,Customer C",
                "vendors": "Vendor X,Vendor Y",
            },
            "customer-success-ops": {
                "team": "customer-success-ops",
                "title": "Customer Success Operations",
                "summary": "Customer communication plans, escalation playbooks, relationship ownership, and SLA complaint handling.",
                "tags": "customer success,escalation,communications,complaint,sla",
                "customers": "Customer A,Customer B,Customer C",
                "vendors": "Vendor X,Vendor Y",
            },
            "platform-operations": {
                "team": "platform-operations",
                "title": "Platform Operations",
                "summary": "Availability incidents, SLO measurement, dependency maps, status updates, and postmortems.",
                "tags": "platform,availability,incident,slo,measurement,vendor dependency",
                "customers": "Customer A,Customer B,Customer C",
                "vendors": "Vendor X,Vendor Y",
            },
            "vendor-risk-management": {
                "team": "vendor-risk-management",
                "title": "Vendor Risk Management",
                "summary": "Vendor tiering, risk registers, contractual obligations, third-party incident escalation, and recovery evidence.",
                "tags": "vendor risk,third party,Vendor X,Vendor Y,contractual rca,evidence",
                "customers": "Customer A,Customer B,Customer C",
                "vendors": "Vendor X,Vendor Y",
            },
            "data-governance": {
                "team": "data-governance",
                "title": "Data Governance",
                "summary": "Audit evidence, data retention, incident metrics lineage, and controlled access to customer impact records.",
                "tags": "audit,evidence,lineage,retention,metrics,governance",
                "customers": "Customer A,Customer B,Customer C",
                "vendors": "Vendor X,Vendor Y",
            },
            "finance": {
                "team": "finance",
                "title": "Finance",
                "summary": "Board-reported revenue metrics, including the Finance definition of ARR used in official financial reporting.",
                "tags": "finance,ARR,board reporting,revenue,recurring",
                "customers": "Acme Singapore,Globex Malaysia,Stark Vietnam",
                "vendors": "",
            },
            "renewals": {
                "team": "renewals",
                "title": "Renewals",
                "summary": "Renewal forecasting metrics, including the Renewals definition of ARR based on committed and likely renewal value.",
                "tags": "renewals,ARR,forecast,committed,likely,revenue",
                "customers": "Acme Singapore,Globex Malaysia,Initech Thailand",
                "vendors": "",
            },
            "sales": {
                "team": "sales",
                "title": "Sales",
                "summary": "Sales and executive presentation metrics, including the Sales definition of ARR based on total booked contract value.",
                "tags": "sales,ARR,bookings,pipeline,presentation,revenue",
                "customers": "Acme Singapore,Hooli Philippines,Stark Vietnam",
                "vendors": "",
            },
        }
        folders = {
            "security-incident-response": {
                "triage-runbooks": "Runbooks for severity classification, first-hour response, and commander handoff.",
                "incident-classification": "Availability, security, privacy, and vendor incident classification guidance.",
                "communications": "Internal and customer-safe incident communications rules.",
            },
            "legal-contracts": {
                "customer-agreements": "Executed and neutralized customer agreement summaries for Customer A, Customer B, and Customer C.",
                "sla-policy": "SLA interpretation, service credit calculation, exclusions, and approval workflow.",
                "notices": "Customer notice templates and legal review rules after availability events.",
            },
            "customer-success-ops": {
                "customer-a": "Customer A account plans, complaint handling, executive contacts, and SLA escalation notes.",
                "customer-playbooks": "Reusable customer success playbooks for availability and vendor-related incidents.",
                "executive-comms": "Executive update templates, tone guidance, and relationship-owner handoffs.",
            },
            "platform-operations": {
                "incidents": "Operational records for availability incidents and dependency failures.",
                "slo-measurement": "Customer availability measurement, exclusion policy, and SLO calculation evidence.",
                "postmortems": "Post-incident reviews, corrective actions, and operational follow-up records.",
            },
            "vendor-risk-management": {
                "vendor-x": "Vendor X tiering, obligations, incident record, and claim evidence.",
                "vendor-y": "Vendor Y reference material and distractor records for unrelated incidents.",
                "risk-registers": "Cross-vendor tiering, review cadence, and dependency risk controls.",
            },
            "data-governance": {
                "audit-evidence": "Evidence packs that connect incident metrics, customer notices, and approval records.",
                "retention": "Retention and legal hold requirements for incident and SLA evidence.",
                "access-control": "Role-based access rules for customer impact data and incident records.",
            },
            "finance": {
                "metric-definitions": "Official Finance metric definitions used for board and statutory reporting.",
            },
            "renewals": {
                "metric-definitions": "Renewals metric definitions used for renewal forecasting and pipeline reviews.",
            },
            "sales": {
                "metric-definitions": "Sales metric definitions used for executive presentations and bookings reviews.",
            },
        }

        def front_matter(fields: dict[str, str]) -> str:
            return "---\n" + "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n---\n"

        def add_metadata(path: str, fields: dict[str, str], heading: str, body: str) -> None:
            if path in files:
                raise ValueError(f"duplicate seed path: {path}")
            files[path] = f"{front_matter(fields)}# {heading}\n\n{body.strip()}\n"

        for repo, meta in repos.items():
            add_metadata(
                f"{repo}/.datameta.md",
                {
                    "id": f"{repo}-repository-metadata",
                    "type": "repository_metadata",
                    "metadata_level": "repository",
                    "repository": repo,
                    "folder": "",
                    "team": meta["team"],
                    "entity": meta["title"],
                    "scope": "repository",
                    "title": meta["title"],
                    "summary": meta["summary"],
                    "customers": meta["customers"],
                    "vendors": meta["vendors"],
                    "slas": "availability-sla",
                    "incidents": "inc-vendor-x-2026-05-20,inc-vendor-y-2026-04-11",
                    "tags": meta["tags"],
                    "neo4j_labels": "Repository",
                    "neo4j_relationships": "[]",
                    "updated_by": "DataMeta Seed",
                    "updated_at": "2026-06-06T00:00:00+00:00",
                },
                meta["title"],
                meta["summary"],
            )
            for folder, summary in folders[repo].items():
                add_metadata(
                    f"{repo}/{folder}/.datameta.md",
                    {
                        "id": f"{repo}-{folder}-folder-metadata",
                        "type": "folder_metadata",
                        "metadata_level": "folder",
                        "repository": repo,
                        "folder": folder,
                        "team": meta["team"],
                        "entity": folder.replace("-", " ").title(),
                        "scope": "folder",
                        "title": folder.replace("-", " ").title(),
                        "summary": summary,
                        "customers": meta["customers"],
                        "vendors": meta["vendors"],
                        "slas": "availability-sla",
                        "incidents": "inc-vendor-x-2026-05-20,inc-vendor-y-2026-04-11",
                        "tags": f"{meta['tags']},{folder.replace('-', ' ')}",
                        "neo4j_labels": "Folder",
                        "neo4j_relationships": "[]",
                        "updated_by": "DataMeta Seed",
                        "updated_at": "2026-06-06T00:00:00+00:00",
                    },
                    folder.replace("-", " ").title(),
                    summary,
                )

        def add_doc(
            repo: str,
            folder: str,
            filename: str,
            *,
            doc_id: str,
            doc_type: str,
            entity: str,
            scope: str,
            title: str,
            summary: str,
            customers: str = "",
            vendors: str = "",
            slas: str = "",
            incidents: str = "",
            tags: str = "",
            required_columns: str = "",
            formula_sql: str = "",
            preferred_tables: str = "",
            body: str,
        ) -> None:
            extra: dict[str, str] = {}
            if required_columns or formula_sql or preferred_tables:
                extra = {
                    "required_columns": required_columns,
                    "formula_sql": formula_sql,
                    "preferred_tables": preferred_tables,
                }
            add_metadata(
                f"{repo}/{folder}/{filename}",
                {
                    "id": doc_id,
                    "type": doc_type,
                    "metadata_level": "file",
                    "repository": repo,
                    "folder": folder,
                    "team": repos[repo]["team"],
                    "entity": entity,
                    "scope": scope,
                    "title": title,
                    "summary": summary,
                    "customers": customers,
                    "vendors": vendors,
                    "slas": slas,
                    "incidents": incidents,
                    "tags": tags,
                    **extra,
                    "neo4j_labels": "Document",
                    "neo4j_relationships": json.dumps(
                        [
                            {"type": "IN_REPOSITORY", "target": repo},
                            {"type": "IN_FOLDER", "target": f"{repo}/{folder}"},
                        ],
                        sort_keys=True,
                    ),
                    "updated_by": "DataMeta Seed",
                    "updated_at": "2026-06-06T00:00:00+00:00",
                },
                title,
                body,
            )

        add_doc(
            "legal-contracts",
            "customer-agreements",
            "customer-a-availability-sla.md",
            doc_id="customer-a-availability-sla",
            doc_type="sla",
            entity="Customer A Availability SLA",
            scope="customer_a_availability",
            title="Customer A Availability SLA",
            summary="Customer A has a 99.90 percent monthly production API availability commitment, with Vendor X outages counting unless specifically excluded.",
            customers="Customer A",
            vendors="Vendor X",
            slas="availability-sla",
            incidents="inc-vendor-x-2026-05-20",
            tags="Customer A,availability SLA,service credit,Vendor X,production API",
            body="""
## Commitment

Customer A's enterprise agreement commits DataMeta to 99.90 percent monthly production API availability.

## Exclusions

Scheduled maintenance approved at least 72 hours in advance is excluded. Third-party vendor outages are excluded only when the vendor is listed in the approved dependency schedule for Customer A. Vendor X is not listed as an excluded dependency for Customer A.

## Service Credit

If monthly production API availability is below 99.90 percent and at least 99.00 percent, Customer A is eligible for a 10 percent service credit for the affected monthly platform fee. If availability is below 99.00 percent, the credit is 25 percent.

## Response Duties

Customer A must receive a preliminary incident explanation within three business days and a final RCA or corrective-action summary within five business days. Customer Success may acknowledge the complaint, but Legal approves service-credit language.
""",
        )
        add_doc(
            "legal-contracts",
            "customer-agreements",
            "customer-b-availability-sla.md",
            doc_id="customer-b-availability-sla",
            doc_type="sla",
            entity="Customer B Availability SLA",
            scope="customer_b_availability",
            title="Customer B Availability SLA",
            summary="Customer B has a 99.50 percent availability target and a broader third-party exclusion schedule.",
            customers="Customer B",
            vendors="Vendor X,Vendor Y",
            slas="availability-sla",
            tags="Customer B,availability SLA,distractor",
            body="Customer B has a 99.50 percent monthly availability target. Vendor X and Vendor Y are listed as excluded dependencies for Customer B unless gross negligence is confirmed.",
        )
        add_doc(
            "legal-contracts",
            "customer-agreements",
            "customer-c-availability-sla.md",
            doc_id="customer-c-availability-sla",
            doc_type="sla",
            entity="Customer C Availability SLA",
            scope="customer_c_availability",
            title="Customer C Availability SLA",
            summary="Customer C has a 99.90 percent availability target but no Vendor X incident in May 2026.",
            customers="Customer C",
            vendors="Vendor Y",
            slas="availability-sla",
            tags="Customer C,availability SLA,distractor",
            body="Customer C has a 99.90 percent availability target. Customer C did not report impact from the Vendor X incident on 2026-05-20.",
        )
        add_doc(
            "legal-contracts",
            "sla-policy",
            "service-credit-approval.md",
            doc_id="service-credit-approval",
            doc_type="policy",
            entity="Service Credit Approval",
            scope="availability_sla",
            title="Service Credit Approval Workflow",
            summary="Legal and Finance must approve service credit offers after availability SLA misses.",
            customers="Customer A,Customer B,Customer C",
            vendors="Vendor X,Vendor Y",
            slas="availability-sla",
            tags="service credit,legal approval,availability",
            body="Customer Success may tell a customer that DataMeta is reviewing SLA eligibility. Legal approves contractual interpretation and Finance approves the credit amount before any concession is offered.",
        )
        add_doc(
            "legal-contracts",
            "notices",
            "availability-incident-notice-template.md",
            doc_id="availability-incident-notice-template",
            doc_type="template",
            entity="Availability Incident Notice",
            scope="customer_notice",
            title="Availability Incident Notice Template",
            summary="Template for customer notices after availability incidents, including non-admission language.",
            customers="Customer A,Customer B,Customer C",
            tags="notice,template,availability,legal review",
            body="Use factual timing, affected services, mitigation, next steps, and review status. Do not concede breach, negligence, or service-credit eligibility until Legal completes review.",
        )

        add_doc(
            "customer-success-ops",
            "customer-a",
            "customer-a-complaint-2026-05.md",
            doc_id="customer-a-complaint-2026-05",
            doc_type="customer_record",
            entity="Customer A SLA Complaint",
            scope="customer_a_escalation",
            title="Customer A SLA Complaint for May 2026",
            summary="Customer A complained that DataMeta missed the availability SLA after the Vendor X incident.",
            customers="Customer A",
            vendors="Vendor X",
            slas="availability-sla",
            incidents="inc-vendor-x-2026-05-20",
            tags="Customer A,complaint,SLA miss,Vendor X",
            body="Customer A's operations lead asked whether the 2026-05-20 Vendor X incident caused DataMeta to miss the May availability SLA. Customer A requested a credit review and final RCA.",
        )
        add_doc(
            "customer-success-ops",
            "customer-a",
            "customer-a-escalation-playbook.md",
            doc_id="customer-a-escalation-playbook",
            doc_type="runbook",
            entity="Customer A Escalation",
            scope="sla_complaint_response",
            title="Customer A SLA Complaint Escalation Playbook",
            summary="Customer Success should acknowledge Customer A within four business hours, gather Legal and Platform evidence, and avoid concession language.",
            customers="Customer A",
            vendors="Vendor X",
            slas="availability-sla",
            incidents="inc-vendor-x-2026-05-20",
            tags="Customer A,escalation,customer success,legal review",
            body="Acknowledge receipt within four business hours. Open a Legal review for SLA interpretation, ask Platform Operations for measured availability and incident timing, ask Vendor Risk for Vendor X recovery evidence, and tell Customer A that service-credit eligibility is under review.",
        )
        add_doc(
            "customer-success-ops",
            "customer-playbooks",
            "availability-sla-complaint-handling.md",
            doc_id="availability-sla-complaint-handling",
            doc_type="runbook",
            entity="Availability SLA Complaint Handling",
            scope="customer_success",
            title="Availability SLA Complaint Handling",
            summary="Customer Success uses a cross-functional review before confirming any SLA miss or credit.",
            customers="Customer A,Customer B,Customer C",
            vendors="Vendor X,Vendor Y",
            slas="availability-sla",
            tags="SLA complaint,customer success,availability",
            body="For any availability SLA complaint, collect the executed SLA, measured monthly availability, incident report, vendor dependency treatment, and approved customer response. Do not promise credits before Legal and Finance approval.",
        )
        add_doc(
            "customer-success-ops",
            "executive-comms",
            "customer-a-executive-update.md",
            doc_id="customer-a-executive-update",
            doc_type="template",
            entity="Customer A Executive Update",
            scope="executive_comms",
            title="Customer A Executive Update Template",
            summary="Executive update template for Customer A after the Vendor X availability incident.",
            customers="Customer A",
            vendors="Vendor X",
            incidents="inc-vendor-x-2026-05-20",
            tags="Customer A,executive communications,Vendor X",
            body="Use concise facts: what happened, what was affected, current recovery state, expected RCA timing, and the fact that SLA and credit review is in progress.",
        )

        add_doc(
            "vendor-risk-management",
            "vendor-x",
            "vendor-x-risk-register.md",
            doc_id="vendor-x-risk-register",
            doc_type="vendor",
            entity="Vendor X",
            scope="risk_register",
            title="Vendor X Risk Register",
            summary="Vendor X is a Tier 1 identity and token validation provider with five-business-day RCA duties after any availability-impacting incident.",
            customers="Customer A,Customer B",
            vendors="Vendor X",
            incidents="inc-vendor-x-2026-05-20",
            tags="Vendor X,Tier 1,RCA,contractual obligation",
            body="Vendor X is Tier 1 because production API authentication depends on its token validation service. After any availability-impacting incident, Vendor X must provide preliminary facts within one business day and a contractual RCA within five business days.",
        )
        add_doc(
            "vendor-risk-management",
            "vendor-x",
            "vendor-x-incident-evidence.md",
            doc_id="vendor-x-incident-evidence",
            doc_type="evidence",
            entity="Vendor X Incident Evidence",
            scope="vendor_claim",
            title="Vendor X Incident Evidence Pack",
            summary="Evidence retained for the Vendor X incident includes outage notice, restoration note, and requested RCA.",
            customers="Customer A",
            vendors="Vendor X",
            incidents="inc-vendor-x-2026-05-20",
            tags="Vendor X,evidence,restoration,RCA",
            body="Vendor Risk retained Vendor X's outage notice at 14:07 UTC, restoration note at 17:30 UTC, and an open request for final RCA. The vendor incident should be linked to Customer A's SLA review but does not by itself decide customer credit eligibility.",
        )
        add_doc(
            "vendor-risk-management",
            "vendor-y",
            "vendor-y-export-delay.md",
            doc_id="vendor-y-export-delay",
            doc_type="vendor",
            entity="Vendor Y",
            scope="risk_register",
            title="Vendor Y Export Delay",
            summary="Vendor Y had a non-production export delay unrelated to Customer A availability.",
            customers="Customer B",
            vendors="Vendor Y",
            incidents="inc-vendor-y-2026-04-11",
            tags="Vendor Y,distractor,export delay",
            body="Vendor Y delayed a billing export for Customer B. It did not affect Customer A, production API availability, or Vendor X token validation.",
        )
        add_doc(
            "vendor-risk-management",
            "risk-registers",
            "tier-one-vendor-controls.md",
            doc_id="tier-one-vendor-controls",
            doc_type="policy",
            entity="Tier One Vendor Controls",
            scope="vendor_risk",
            title="Tier One Vendor Controls",
            summary="Tier 1 vendors require incident notifications, annual resilience review, and customer-impact evidence retention.",
            vendors="Vendor X,Vendor Y",
            tags="vendor risk,Tier 1,incident notification,evidence",
            body="For Tier 1 vendors, open a vendor incident record, capture notices and RCA artifacts, and confirm whether customer contracts exclude the vendor dependency.",
        )

        add_doc(
            "security-incident-response",
            "triage-runbooks",
            "vendor-outage-triage.md",
            doc_id="vendor-outage-triage",
            doc_type="runbook",
            entity="Vendor Outage Triage",
            scope="incident_response",
            title="Vendor Outage Triage Runbook",
            summary="Security Incident Response classifies vendor outages and checks whether the event is availability-only or includes security impact.",
            customers="Customer A,Customer B,Customer C",
            vendors="Vendor X,Vendor Y",
            incidents="inc-vendor-x-2026-05-20",
            tags="incident response,vendor outage,classification",
            body="Classify the incident, assign an incident commander, verify whether customer data confidentiality or integrity is affected, and hand off availability measurement to Platform Operations. The Vendor X incident is availability-only unless new evidence indicates data exposure.",
        )
        add_doc(
            "security-incident-response",
            "incident-classification",
            "availability-vs-security-classification.md",
            doc_id="availability-vs-security-classification",
            doc_type="policy",
            entity="Incident Classification",
            scope="availability_security",
            title="Availability Versus Security Classification",
            summary="Availability incidents require security review only when confidentiality, integrity, or regulated reporting triggers appear.",
            vendors="Vendor X,Vendor Y",
            tags="classification,availability,security",
            body="A vendor outage that causes failed authentication may remain an availability incident if no unauthorized access, disclosure, or data integrity risk is found.",
        )
        add_doc(
            "security-incident-response",
            "communications",
            "incident-comms-approval.md",
            doc_id="incident-comms-approval",
            doc_type="policy",
            entity="Incident Communications Approval",
            scope="incident_comms",
            title="Incident Communications Approval",
            summary="Customer-facing incident communications require Customer Success ownership and Legal review when SLA or credit language appears.",
            customers="Customer A,Customer B,Customer C",
            tags="communications,legal review,customer success",
            body="Security Incident Response can provide facts and timing. Customer Success owns the customer message, and Legal must review any SLA miss, breach, or credit language.",
        )

        add_doc(
            "data-governance",
            "audit-evidence",
            "customer-a-sla-evidence-checklist.md",
            doc_id="customer-a-sla-evidence-checklist",
            doc_type="evidence",
            entity="Customer A SLA Evidence Checklist",
            scope="audit_evidence",
            title="Customer A SLA Evidence Checklist",
            summary="Evidence checklist for Customer A's SLA review includes the contract, incident timeline, availability calculation, notices, and vendor artifacts.",
            customers="Customer A",
            vendors="Vendor X",
            slas="availability-sla",
            incidents="inc-vendor-x-2026-05-20",
            tags="Customer A,audit evidence,SLA,Vendor X",
            body="Retain Customer A's executed SLA, Platform Operations availability measurement, incident event timeline, Customer Success complaint record, Legal approval record, and Vendor X notice and RCA artifacts.",
        )
        add_doc(
            "data-governance",
            "retention",
            "incident-retention-policy.md",
            doc_id="incident-retention-policy",
            doc_type="policy",
            entity="Incident Retention Policy",
            scope="retention",
            title="Incident Retention Policy",
            summary="Customer-impacting availability incident evidence is retained for seven years when tied to a contractual SLA complaint.",
            customers="Customer A,Customer B,Customer C",
            vendors="Vendor X,Vendor Y",
            tags="retention,incident evidence,SLA complaint",
            body="When a customer raises an SLA complaint, preserve incident logs, calculations, customer notices, vendor communications, and approval records for seven years or longer if Legal places a hold.",
        )
        add_doc(
            "data-governance",
            "access-control",
            "customer-impact-access.md",
            doc_id="customer-impact-access",
            doc_type="policy",
            entity="Customer Impact Access",
            scope="access_control",
            title="Customer Impact Data Access",
            summary="Customer-impact data can be read by incident, legal, platform, customer success, vendor risk, and data governance roles for the affected case.",
            customers="Customer A,Customer B,Customer C",
            tags="access control,customer impact,RBAC",
            body="Customer-impact details are need-to-know. Cross-functional reviewers may access the affected customer records while the SLA review is open.",
        )

        # --- Single-repo anchor docs ---------------------------------------
        # Each of these is self-contained so a question about it resolves to a
        # single repository, making the single-repo demo questions obvious.
        add_doc(
            "security-incident-response",
            "incident-classification",
            "severity-levels.md",
            doc_id="incident-severity-levels",
            doc_type="policy",
            entity="Incident Severity Levels",
            scope="severity_classification",
            title="Incident Severity Levels",
            summary="DataMeta classifies incidents from Sev-1 (critical, full outage) to Sev-4 (low), each with a target response and update cadence.",
            incidents="inc-vendor-x-2026-05-20",
            tags="severity,Sev-1,Sev-2,Sev-3,Sev-4,classification,response time",
            body="""
## Severity Matrix

- **Sev-1 - Critical:** Full production outage or a confirmed data breach. Page the on-call commander immediately. Customer updates every 30 minutes.
- **Sev-2 - Major:** Significant degradation, or a single large customer impacted. Respond within 30 minutes. Updates every hour.
- **Sev-3 - Minor:** Limited or intermittent impact with a workaround. Respond within four business hours.
- **Sev-4 - Low:** Cosmetic or internal-only issue. Handle during normal working hours.

The Vendor X outage on 2026-05-20 was declared Sev-2 because it degraded production availability for a single enterprise customer.
""",
        )
        add_doc(
            "legal-contracts",
            "sla-policy",
            "service-credit-schedule.md",
            doc_id="service-credit-schedule",
            doc_type="policy",
            entity="Service Credit Schedule",
            scope="availability_sla",
            title="Service Credit Schedule",
            summary="The standard availability service-credit schedule: a 10 percent credit below 99.90 percent, and 25 percent below 99.00 percent.",
            customers="Customer A,Customer B,Customer C",
            slas="availability-sla",
            tags="service credit,schedule,availability,10 percent,25 percent",
            body="""
## Standard Service Credit Schedule

| Monthly production API availability | Service credit |
| --- | --- |
| Below 99.90% and at or above 99.00% | 10% of the monthly platform fee |
| Below 99.00% | 25% of the monthly platform fee |

Credits apply to the affected month's platform fee. Legal approves the contractual interpretation and Finance approves the amount before any credit is offered to a customer.
""",
        )
        add_doc(
            "data-governance",
            "retention",
            "evidence-retention.md",
            doc_id="evidence-retention",
            doc_type="policy",
            entity="Incident Evidence Retention",
            scope="retention",
            title="Incident Evidence Retention",
            summary="Incident and SLA evidence is retained for seven years; an active legal hold suspends deletion until it is released in writing.",
            tags="retention,legal hold,evidence,seven years",
            body="""
## Retention Periods

- Incident records, postmortems, and availability evidence: **7 years**.
- Customer notices and service-credit approvals: **7 years**.
- Routine operational logs: **13 months**.

## Legal Hold

When Legal places a hold on a case, all related records are preserved and exempt from scheduled deletion until the hold is released in writing.
""",
        )
        add_doc(
            "vendor-risk-management",
            "risk-registers",
            "vendor-tiering.md",
            doc_id="vendor-tiering",
            doc_type="policy",
            entity="Vendor Tiering",
            scope="vendor_tiering",
            title="Vendor Tiering",
            summary="Vendors are tiered 1 to 3 by how critical they are to production. Vendor X is Tier 1 (critical); Vendor Y is Tier 3.",
            vendors="Vendor X,Vendor Y",
            tags="vendor tiering,Tier 1,Tier 3,Vendor X,Vendor Y,critical",
            body="""
## Tiers

- **Tier 1 - Critical:** An outage of this vendor directly degrades customer availability. Reviewed quarterly. **Vendor X** is Tier 1.
- **Tier 2 - Important:** Supports production but has a fallback. Reviewed twice a year.
- **Tier 3 - Standard:** Internal or non-production tooling. Reviewed annually. **Vendor Y** is Tier 3.

Tier 1 vendors must provide a root-cause analysis within five business days of any incident.
""",
        )
        add_doc(
            "platform-operations",
            "slo-measurement",
            "availability-slo.md",
            doc_id="availability-slo",
            doc_type="policy",
            entity="Availability SLO Measurement",
            scope="slo_measurement",
            title="Availability SLO Measurement",
            summary="Customer availability is measured monthly as available production API minutes over total minutes, targeting 99.90 percent.",
            customers="Customer A,Customer B,Customer C",
            slas="availability-sla",
            tags="SLO,availability,measurement,99.90 percent,uptime",
            body="""
## How We Measure

Monthly availability = (total production API minutes - unavailable minutes) / total minutes, calculated per customer.

The internal SLO target is **99.90%**. Approved maintenance windows are excluded; third-party vendor outages are excluded only when the vendor is an approved excluded dependency for that customer.

Measurements are the system of record for any SLA review and are stored in the availability evidence table.
""",
        )
        add_doc(
            "customer-success-ops",
            "customer-a",
            "customer-a-account-profile.md",
            doc_id="customer-a-account-profile",
            doc_type="profile",
            entity="Customer A Account Profile",
            scope="account_profile",
            title="Customer A Account Profile",
            summary="Customer A is a Tier 1 enterprise account owned by relationship manager Morgan Lee, on the 99.90 percent availability plan.",
            customers="Customer A",
            tags="Customer A,account profile,relationship owner,Morgan Lee,Tier 1",
            body="""
## Account Profile

- **Account tier:** Tier 1 enterprise
- **Relationship owner:** Morgan Lee, Senior Customer Success Manager
- **Executive sponsor:** VP of Customer Success
- **Availability plan:** 99.90% monthly production API
- **Primary contact:** Customer A IT Operations

Escalations for Customer A route first to Morgan Lee, then to the executive sponsor.
""",
        )

        # --- Multi-repo storyline: Vendor X outage on 2026-05-20 -----------
        # These docs live in different repos but all tie to the same incident,
        # so the multi-repo demo question pulls a traceable chain of evidence.
        add_doc(
            "platform-operations",
            "incidents",
            "vendor-x-outage-2026-05-20.md",
            doc_id="vendor-x-outage-2026-05-20",
            doc_type="incident",
            entity="Vendor X Outage 2026-05-20",
            scope="availability_incident",
            title="Vendor X Outage 2026-05-20 Incident Record",
            summary="A Vendor X regional outage on 2026-05-20 degraded Customer A production API availability for about three hours; declared Sev-2.",
            customers="Customer A",
            vendors="Vendor X",
            slas="availability-sla",
            incidents="inc-vendor-x-2026-05-20",
            tags="incident,Vendor X,Customer A,availability,Sev-2,2026-05-20",
            body="""
## Summary

On 2026-05-20 a Vendor X regional outage degraded DataMeta's production API for Customer A between 09:12 and 12:20 UTC (about 3 hours 8 minutes). Declared **Sev-2**.

## Impact

Customer A's production API returned elevated errors during the window. Customer B and Customer C were not impacted.

## Timeline

- 09:12 - Error-rate alert fires; on-call paged.
- 09:25 - Vendor X confirmed as the source.
- 10:05 - Customer A relationship owner notified.
- 12:20 - Vendor X recovers; service restored.

This incident is the basis for Customer A's May 2026 SLA complaint.
""",
        )
        add_doc(
            "platform-operations",
            "postmortems",
            "vendor-x-outage-postmortem.md",
            doc_id="vendor-x-outage-postmortem",
            doc_type="postmortem",
            entity="Vendor X Outage Postmortem",
            scope="postmortem",
            title="Vendor X Outage Postmortem",
            summary="Root cause was a Vendor X regional failover defect. Corrective actions: multi-region failover and a required Vendor X RCA.",
            customers="Customer A",
            vendors="Vendor X",
            incidents="inc-vendor-x-2026-05-20",
            tags="postmortem,Vendor X,root cause,corrective action,failover",
            body="""
## Root Cause

Vendor X's regional failover did not trigger, leaving requests routed to a failed region for the duration of the outage.

## Corrective Actions

1. Add automated multi-region failover for the Vendor X dependency (owner: Platform Ops).
2. Require a formal Vendor X RCA within five business days (owner: Vendor Risk).
3. Confirm Customer A's measured availability and evidence pack with Data Governance.

## Customer Impact

Customer A only. See the Customer A May 2026 availability measurement for the exact figure.
""",
        )
        add_doc(
            "platform-operations",
            "slo-measurement",
            "customer-a-may-2026-availability.md",
            doc_id="customer-a-may-2026-availability",
            doc_type="measurement",
            entity="Customer A May 2026 Availability",
            scope="availability_measurement",
            title="Customer A May 2026 Availability",
            summary="Customer A's measured production API availability for May 2026 was 99.62 percent, below the 99.90 percent commitment.",
            customers="Customer A",
            slas="availability-sla",
            incidents="inc-vendor-x-2026-05-20",
            tags="measurement,Customer A,availability,99.62 percent,May 2026",
            body="""
## Result

Customer A production API availability for May 2026: **99.62%**.

This is below the 99.90% commitment and at or above 99.00%, which maps to a **10% service credit** under the standard schedule. The shortfall is attributable to the Vendor X outage on 2026-05-20. Vendor X is **not** an approved excluded dependency for Customer A, so the downtime counts against the SLA.
""",
        )
        add_doc(
            "vendor-risk-management",
            "vendor-x",
            "vendor-x-incident-rca.md",
            doc_id="vendor-x-incident-rca",
            doc_type="rca",
            entity="Vendor X Incident RCA",
            scope="vendor_rca",
            title="Vendor X Incident RCA",
            summary="Vendor X delivered a root-cause analysis for the 2026-05-20 outage confirming a failover defect; remediation is in progress.",
            vendors="Vendor X",
            incidents="inc-vendor-x-2026-05-20",
            tags="RCA,Vendor X,failover defect,Tier 1,evidence",
            body="""
## Vendor X RCA

Vendor X confirmed the 2026-05-20 outage was caused by a defect in their regional failover automation. Vendor X has committed to a fix and provided this RCA within the five-business-day window required for Tier 1 vendors.

This RCA is the evidence required before DataMeta finalizes Customer A's service-credit decision.
""",
        )
        add_doc(
            "security-incident-response",
            "communications",
            "customer-a-incident-notice.md",
            doc_id="customer-a-incident-notice",
            doc_type="notice",
            entity="Customer A Incident Notice",
            scope="customer_communication",
            title="Customer A Incident Notice",
            summary="Customer-safe notice to Customer A acknowledging the 2026-05-20 availability impact without stating SLA liability.",
            customers="Customer A",
            vendors="Vendor X",
            incidents="inc-vendor-x-2026-05-20",
            tags="notice,Customer A,communication,customer-safe",
            body="""
## Customer-Safe Notice

We confirmed a service disruption affecting your production API on 2026-05-20 caused by a third-party dependency. Our team restored service the same day and is completing a full review.

Notices acknowledge impact and next steps. They do **not** state SLA eligibility or offer credits - Legal owns that language.
""",
        )
        add_doc(
            "data-governance",
            "audit-evidence",
            "customer-a-may-2026-evidence-pack.md",
            doc_id="customer-a-may-2026-evidence-pack",
            doc_type="evidence_pack",
            entity="Customer A May 2026 Evidence Pack",
            scope="audit_evidence",
            title="Customer A May 2026 Evidence Pack",
            summary="Audit evidence pack linking the Vendor X incident, Customer A's 99.62 percent measured availability, the customer notice, and the Vendor X RCA.",
            customers="Customer A",
            vendors="Vendor X",
            slas="availability-sla",
            incidents="inc-vendor-x-2026-05-20",
            tags="evidence pack,audit,Customer A,Vendor X,service credit",
            body="""
## Evidence Pack Contents

- Incident record: Vendor X outage 2026-05-20 (Sev-2).
- Measured availability: Customer A 99.62% for May 2026.
- Customer notice: sent 2026-05-20.
- Vendor X RCA: received within the Tier 1 window.
- Postmortem: multi-region failover corrective action.

This pack supports the service-credit decision and is retained for seven years under the incident evidence retention policy.
""",
        )

        # --- Conflicting ARR definitions across teams -----------------------
        # Three teams define ARR differently. When an operations user who can
        # read all three asks "calculate ARR", the app must pause and ask which
        # definition to use instead of guessing. Each formula runs against the
        # arr_subscriptions table and returns a different number for ASEAN.
        add_doc(
            "finance",
            "metric-definitions",
            "arr-definition.md",
            doc_id="arr-finance-board",
            doc_type="definition",
            entity="ARR",
            scope="board_reporting",
            title="ARR (Finance, Board Reporting)",
            summary="Finance defines ARR as active monthly recurring revenue annualized: sum of MRR for active subscriptions multiplied by 12. Used for board and statutory reporting.",
            tags="ARR,finance,board reporting,MRR,active,annualized",
            required_columns="monthly_recurring_revenue,status,region",
            formula_sql="SELECT SUM(monthly_recurring_revenue) * 12 FROM {table} WHERE region = 'ASEAN' AND status = 'active'",
            preferred_tables="arr_subscriptions",
            body="""
## Finance Definition of ARR

Finance reports ARR as **active monthly recurring revenue, annualized**:

- Include only subscriptions with `status = 'active'`.
- Sum `monthly_recurring_revenue` and multiply by 12.
- Exclude churned subscriptions and any contract value not yet recognized as recurring revenue.

This is the figure used in board decks and statutory reporting.
""",
        )
        add_doc(
            "renewals",
            "metric-definitions",
            "arr-definition.md",
            doc_id="arr-renewals-forecast",
            doc_type="definition",
            entity="ARR",
            scope="renewal_forecast",
            title="ARR (Renewals, Forecast)",
            summary="Renewals defines ARR as the annual contract value of committed and likely renewals. Already annual, so it is not multiplied by 12.",
            tags="ARR,renewals,forecast,committed,likely,annual contract value",
            required_columns="annual_contract_value,renewal_status,region",
            formula_sql="SELECT SUM(annual_contract_value) FROM {table} WHERE region = 'ASEAN' AND renewal_status IN ('committed', 'likely')",
            preferred_tables="arr_subscriptions",
            body="""
## Renewals Definition of ARR

Renewals forecasts ARR from the **annual contract value of renewals it expects to close**:

- Include subscriptions where `renewal_status` is `committed` or `likely`.
- Sum `annual_contract_value` directly (these values are already annual - do not multiply by 12).
- Exclude pure pipeline and lost renewals.

This is the figure used in renewal forecasts and pipeline reviews.
""",
        )
        add_doc(
            "sales",
            "metric-definitions",
            "arr-definition.md",
            doc_id="arr-sales-presentation",
            doc_type="definition",
            entity="ARR",
            scope="exec_presentation",
            title="ARR (Sales, Executive Presentation)",
            summary="Sales presents ARR as total booked annual contract value across all non-churned accounts, including pipeline. The most optimistic figure, used in executive presentations.",
            tags="ARR,sales,bookings,pipeline,presentation,annual contract value",
            required_columns="annual_contract_value,status,region",
            formula_sql="SELECT SUM(annual_contract_value) FROM {table} WHERE region = 'ASEAN' AND status != 'churned'",
            preferred_tables="arr_subscriptions",
            body="""
## Sales Definition of ARR

For executive presentations, Sales reports ARR as **total booked annual contract value**:

- Include every subscription that is not `churned`, including new logos still in `pipeline`.
- Sum `annual_contract_value`.
- This is the broadest, most optimistic figure - it is what the area head typically asks for in presentations.

Because it counts pipeline and full booked value, it is higher than the Finance and Renewals figures.
""",
        )

        # --- Realistic filler for any folder still under three file docs ----
        for repo, folder_map in folders.items():
            for folder, folder_summary in folder_map.items():
                existing = [path for path in files if path.startswith(f"{repo}/{folder}/") and not path.endswith(".datameta.md")]
                for index in range(len(existing), 3):
                    customer = ["Customer A", "Customer B", "Customer C"][index % 3]
                    vendor = ["Vendor X", "Vendor Y", "Vendor Y"][index % 3]
                    label = folder.replace("-", " ")
                    add_doc(
                        repo,
                        folder,
                        f"working-note-{index + 1}.md",
                        doc_id=f"{repo}-{folder}-note-{index + 1}",
                        doc_type="note",
                        entity=f"{label.title()} Working Note {index + 1}",
                        scope="reference",
                        title=f"{label.title()} Working Note {index + 1}",
                        summary=f"Supporting working note for {label} referencing {customer} and {vendor}; background detail only, not an authoritative SLA or incident record.",
                        customers=customer,
                        vendors=vendor,
                        tags=f"{label},working note,{customer},{vendor}",
                        body=(
                            f"## {label.title()} Working Note\n\n"
                            f"This note captures supporting detail for {label} in the {repos[repo]['title']} repository. "
                            f"It references {customer} and {vendor} for context.\n\n"
                            f"Folder scope: {folder_summary}\n\n"
                            "It is background only and does not override executed SLA terms, measured availability, or formal incident evidence."
                        ),
                    )
        return files

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
        parts = relative_path.split("/")
        repository = metadata.get("repository") or (parts[0] if parts else "")
        inferred_folder = "/".join(parts[1:-1]) if len(parts) > 2 else ""
        folder = metadata.get("folder") or inferred_folder
        if path.name == ".datameta.md":
            metadata_level = metadata.get("metadata_level") or ("folder" if folder else "repository")
        else:
            metadata_level = metadata.get("metadata_level") or "file"
        commit = self._file_commit(relative_path)
        return {
            "path": relative_path,
            "repository": repository,
            "folder": folder,
            "namespace": "/".join(part for part in [repository, folder] if part),
            "metadata_level": metadata_level,
            "metadata": metadata,
            "body": body,
            "text": text,
            "commit": commit,
            "id": metadata.get("id") or slugify(relative_path),
            "team": metadata.get("team") or repository or relative_path.split("/", 1)[0],
            "type": metadata.get("type", "note"),
            "entity": metadata.get("entity", ""),
            "scope": metadata.get("scope", ""),
            "title": metadata.get("title") or relative_path,
            "summary": metadata.get("summary") or body.splitlines()[0].lstrip("# ").strip(),
            "customers": parse_csv(metadata.get("customers")),
            "vendors": parse_csv(metadata.get("vendors")),
            "slas": parse_csv(metadata.get("slas")),
            "incidents": parse_csv(metadata.get("incidents")),
            "tags": parse_csv(metadata.get("tags")),
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
        # Only surface calculation definitions when the question actually names
        # the metric. Otherwise this is a normal knowledge question for the RAG
        # flow, not a calculation that needs a definition chosen.
        for doc in self.visible_documents(user) if wants_arr else []:
            if doc["type"] != "definition":
                continue
            if doc["entity"].lower() != "arr":
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
        if doc["id"] == "arr-sales-presentation":
            return [
                "Use total booked annual contract value.",
                "Include pipeline and new logos, exclude only churned accounts.",
                "Broadest figure - used by the area head in presentations.",
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
            "legal-contracts"
            if "sla" in text or "contract" in text or "credit" in text or "legal" in text
            else "vendor-risk-management"
            if "vendor" in text
            else "customer-success-ops"
            if "customer" in text or "complaint" in text or "communication" in text
            else "platform-operations"
            if "incident" in text or "availability" in text or "outage" in text
            else "data-governance"
            if "evidence" in text or "audit" in text or "retention" in text
            else "security-incident-response"
        )
        entity = (
            "Customer A Availability SLA"
            if "customer a" in text and "sla" in text
            else "Vendor X Incident"
            if "vendor x" in text
            else "Incident Knowledge Note"
        )
        scope = (
            "customer_a_availability"
            if "customer a" in text
            else "vendor_x_availability"
            if "vendor x" in text
            else "general"
        )
        title = f"{entity} for {scope.replace('_', ' ')}"
        formula_sql = ""
        required_columns: list[str] = []
        preferred_tables: list[str] = []
        path = f"{team}/proposed-updates/{slugify(entity)}-{slugify(scope)}.md"
        return {
            "id": f"{slugify(entity)}-{team}-{slugify(scope)}",
            "type": "policy" if "sla" in text or "credit" in text else "note",
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
            "neo4j_labels": ["Document", "Policy"] if "sla" in text or "credit" in text else ["Document", "Note"],
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
                        "entity": doc["entity"],
                        "scope": doc["scope"],
                        "title": doc["title"],
                        "summary": doc["summary"],
                        "markdown": doc["text"],
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
            f"{user.id}@generic-enterprise.local",
        )
        doc = self.parse_document(path)
        neo4j_result = self.sync_document_to_neo4j(doc)
        self._multirepo_index = None
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

    def _get_neo4j_driver(self):
        url = configured_env_value("DATAMETA_NEO4J_URL") or ""
        user = configured_env_value("DATAMETA_NEO4J_USER") or ""
        password = configured_env_value("DATAMETA_NEO4J_PASSWORD") or ""
        key = (url, user, password)
        if self._neo4j_driver is not None and self._neo4j_driver_key == key:
            return self._neo4j_driver
        if self._neo4j_driver is not None:
            try:
                self._neo4j_driver.close()
            except Exception:
                pass
            self._neo4j_driver = None
        from neo4j import GraphDatabase

        self._neo4j_driver = GraphDatabase.driver(url, auth=(user, password))
        self._neo4j_driver_key = key
        return self._neo4j_driver

    def _neo4j_read(self, statement: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]] | None:
        if not self._neo4j_status()["configured"]:
            return None
        try:
            from neo4j.exceptions import DriverError, Neo4jError
        except ImportError:
            return None
        try:
            driver = self._get_neo4j_driver()
            with driver.session(database=neo4j_database()) as session:
                return session.execute_read(
                    lambda tx: [record.data() for record in tx.run(statement, parameters or {})]
                )
        except (DriverError, Neo4jError, Exception):
            return None

    def _run_neo4j_statements(self, statements: list[dict[str, Any]], batch_size: int = 25) -> dict[str, Any]:
        if not statements:
            return {"ok": True, "status": "synced", "errors": []}
        try:
            from neo4j.exceptions import DriverError, Neo4jError
        except ImportError as error:
            return {"ok": False, "status": "driver_missing", "error": str(error)}
        database = neo4j_database()
        try:
            driver = self._get_neo4j_driver()
        except Exception as error:  # configuration / driver construction failure
            return {"ok": False, "status": "sync_failed", "error": str(error)}
        errors: list[dict[str, Any]] = []
        try:
            with driver.session(database=database) as session:
                # Batch into managed write transactions so each chunk is atomic and
                # the free-tier instance is not asked to hold every embedding at once.
                for start in range(0, len(statements), batch_size):
                    batch = statements[start : start + batch_size]

                    def _apply(tx, batch=batch):
                        for stmt in batch:
                            tx.run(stmt["statement"], stmt.get("parameters") or {})

                    try:
                        session.execute_write(_apply)
                    except Neo4jError as error:
                        errors.append({"code": error.code, "message": error.message})
        except DriverError as error:
            return {"ok": False, "status": "sync_failed", "error": str(error)}
        return {"ok": not errors, "status": "synced" if not errors else "sync_failed", "errors": errors}

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
        return self._run_neo4j_statements(statements)

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

    def _metadata_text(self, doc: dict[str, Any]) -> str:
        fields = [
            doc.get("repository", ""),
            doc.get("folder", ""),
            doc.get("team", ""),
            doc.get("type", ""),
            doc.get("entity", ""),
            doc.get("scope", ""),
            doc.get("title", ""),
            doc.get("summary", ""),
            " ".join(doc.get("customers", [])),
            " ".join(doc.get("vendors", [])),
            " ".join(doc.get("slas", [])),
            " ".join(doc.get("incidents", [])),
            " ".join(doc.get("tags", [])),
        ]
        return " ".join(str(field) for field in fields if field)

    def _required_metadata_missing(self, doc: dict[str, Any]) -> list[str]:
        required = ["id", "type", "metadata_level", "repository", "team", "title", "summary", "entity", "scope"]
        if doc["metadata_level"] in {"folder", "file"}:
            required.append("folder")
        missing = []
        for key in required:
            value = doc.get(key) if key in doc else doc["metadata"].get(key)
            if value in {None, ""}:
                missing.append(key)
        return missing

    def _local_embedding(self, text: str) -> list[float]:
        vector = [0.0] * LOCAL_EMBEDDING_DIMENSIONS
        counts: dict[str, int] = {}
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % LOCAL_EMBEDDING_DIMENSIONS
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]

    def _embed_texts(self, texts: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
        config = model_config()
        api_key = configured_env_value("OPENAI_API_KEY")
        provider = configured_env_value("DATAMETA_EMBEDDING_PROVIDER")
        if api_key and config.get("embedding") and provider != "local":
            try:
                if self.openai_client and hasattr(self.openai_client, "embed_texts"):
                    embeddings = self.openai_client.embed_texts(texts, config["embedding"])  # type: ignore[attr-defined]
                else:
                    embeddings = OpenAIResponsesClient(api_key=api_key, model=config["reasoning"]).embed_texts(texts, config["embedding"])
                return embeddings, {
                    "provider": "openai",
                    "model": config["embedding"],
                    "dimensions": len(embeddings[0]) if embeddings else 0,
                    "count": len(embeddings),
                }
            except Exception as error:
                if configured_env_value("DATAMETA_STRICT_OPENAI_EMBEDDINGS") == "1":
                    raise
                return [self._local_embedding(text) for text in texts], {
                    "provider": "local_hash_after_openai_error",
                    "model": config["embedding"],
                    "dimensions": LOCAL_EMBEDDING_DIMENSIONS,
                    "count": len(texts),
                    "error": str(error),
                }
        return [self._local_embedding(text) for text in texts], {
            "provider": "local_hash",
            "model": config.get("embedding"),
            "dimensions": LOCAL_EMBEDDING_DIMENSIONS,
            "count": len(texts),
        }

    def _cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        return sum(a * b for a, b in zip(left, right))

    def _chunk_body(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        chunks = []
        current_heading = doc["title"]
        current_lines: list[str] = []
        for line in doc["body"].splitlines():
            if line.startswith("#"):
                if current_lines:
                    chunks.append(
                        {
                            "heading": current_heading,
                            "text": "\n".join(current_lines).strip(),
                        }
                    )
                    current_lines = []
                current_heading = line.lstrip("# ").strip() or doc["title"]
            else:
                current_lines.append(line)
        if current_lines:
            chunks.append({"heading": current_heading, "text": "\n".join(current_lines).strip()})
        return [
            {
                "id": f"{doc['id']}::chunk-{index + 1}",
                "document_id": doc["id"],
                "path": doc["path"],
                "repository": doc["repository"],
                "folder": doc["folder"],
                "heading": chunk["heading"],
                "text": chunk["text"],
            }
            for index, chunk in enumerate(chunks)
            if chunk["text"]
        ]

    def _neo4j_status(self) -> dict[str, Any]:
        return {
            "required": True,
            "configured": bool(
                configured_env_value("DATAMETA_NEO4J_URL")
                and configured_env_value("DATAMETA_NEO4J_USER")
                and configured_env_value("DATAMETA_NEO4J_PASSWORD")
            ),
            "database": neo4j_database(),
            "transport": "bolt",
            "env": {
                "url": "DATAMETA_NEO4J_URL",
                "user": "DATAMETA_NEO4J_USER",
                "password": "DATAMETA_NEO4J_PASSWORD",
                "database": "DATAMETA_NEO4J_DATABASE",
            },
        }

    def _neo4j_vector_dimensions(self) -> int:
        raw = configured_env_value("DATAMETA_NEO4J_VECTOR_DIMENSIONS")
        if raw and raw.isdigit():
            return int(raw)
        return 3072

    def datameta_index_repos(self, force: bool = False, sync_neo4j: bool = False) -> dict[str, Any]:
        return self.index_repos(force=force, sync_neo4j=sync_neo4j)

    def index_repos(self, force: bool = False, sync_neo4j: bool = False) -> dict[str, Any]:
        self.ensure_ready()
        if self._multirepo_index and not force:
            return self._index_summary(self._multirepo_index)
        docs = self.all_documents()
        repo_nodes = [doc for doc in docs if doc["metadata_level"] == "repository"]
        folder_nodes = [doc for doc in docs if doc["metadata_level"] == "folder"]
        file_nodes = [doc for doc in docs if doc["metadata_level"] == "file"]
        metadata_items: list[dict[str, Any]] = []
        for level, nodes in (("repository", repo_nodes), ("folder", folder_nodes), ("file", file_nodes)):
            for doc in nodes:
                metadata_items.append(
                    {
                        "id": doc["id"],
                        "level": level,
                        "repository": doc["repository"],
                        "folder": doc["folder"],
                        "path": doc["path"],
                        "title": doc["title"],
                        "summary": doc["summary"],
                        "metadata_text": self._metadata_text(doc),
                        "doc": doc,
                    }
                )
        embeddings, embedding_status = self._embed_texts([item["metadata_text"] for item in metadata_items])
        for item, embedding in zip(metadata_items, embeddings):
            item["embedding"] = embedding
            item["tokens"] = tokenize(item["metadata_text"])
        chunk_nodes = []
        for doc in file_nodes:
            chunk_nodes.extend(self._chunk_body(doc))
        entities = {
            "customers": sorted({customer for doc in docs for customer in doc.get("customers", [])}),
            "vendors": sorted({vendor for doc in docs for vendor in doc.get("vendors", [])}),
            "slas": sorted({sla for doc in docs for sla in doc.get("slas", [])}),
            "incidents": sorted({incident for doc in docs for incident in doc.get("incidents", [])}),
            "teams": sorted({doc["team"] for doc in docs if doc.get("team")}),
        }
        index = {
            "indexed_at": utc_now(),
            "repositories": {item["repository"]: item for item in metadata_items if item["level"] == "repository"},
            "folders": {f"{item['repository']}/{item['folder']}": item for item in metadata_items if item["level"] == "folder"},
            "files": {item["path"]: item for item in metadata_items if item["level"] == "file"},
            "chunks": chunk_nodes,
            "entities": entities,
            "embedding": embedding_status,
            "neo4j": self._neo4j_status(),
            "metadata_completeness": [
                {"path": doc["path"], "level": doc["metadata_level"], "missing": self._required_metadata_missing(doc)}
                for doc in docs
                if self._required_metadata_missing(doc)
            ],
        }
        if sync_neo4j:
            index["neo4j_sync"] = self._sync_multirepo_index_to_neo4j(index)
        self._multirepo_index = index
        return self._index_summary(index)

    def _index_summary(self, index: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": not index["metadata_completeness"],
            "indexed_at": index["indexed_at"],
            "counts": {
                "repositories": len(index["repositories"]),
                "folders": len(index["folders"]),
                "files": len(index["files"]),
                "chunks": len(index["chunks"]),
                "customers": len(index["entities"]["customers"]),
                "vendors": len(index["entities"]["vendors"]),
                "slas": len(index["entities"]["slas"]),
                "incidents": len(index["entities"]["incidents"]),
                "teams": len(index["entities"]["teams"]),
            },
            "embedding": index["embedding"],
            "neo4j": index["neo4j"],
            "neo4j_sync": index.get("neo4j_sync"),
            "metadata_completeness": index["metadata_completeness"],
        }

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
        # The markdown repository is the source of truth: prune synced nodes
        # whose source no longer exists so deletions propagate to Neo4j.
        for label, ids in (
            ("Repository", [item["id"] for item in index["repositories"].values()]),
            ("Folder", [item["id"] for item in index["folders"].values()]),
            ("Document", [item["id"] for item in index["files"].values()]),
            ("Chunk", [chunk["id"] for chunk in index["chunks"]]),
        ):
            statements.append(
                {
                    "statement": f"MATCH (n:{label}) WHERE NOT n.id IN $ids DETACH DELETE n",
                    "parameters": {"ids": ids},
                }
            )
        return statements

    def _sync_multirepo_index_to_neo4j(self, index: dict[str, Any]) -> dict[str, Any]:
        neo4j = self._neo4j_status()
        if not neo4j["configured"]:
            return {"ok": False, "status": "not_configured", "required": True}
        statements = self._build_neo4j_sync_statements(index)
        return self._run_neo4j_statements(statements)

    def datameta_repo_inventory(self) -> dict[str, Any]:
        return self.repo_inventory()

    def repo_inventory(self) -> dict[str, Any]:
        self.index_repos()
        assert self._multirepo_index is not None
        index = self._multirepo_index
        repos = []
        for repo, item in sorted(index["repositories"].items()):
            folders = []
            for folder_key, folder_item in sorted(index["folders"].items()):
                if folder_item["repository"] != repo:
                    continue
                files = [
                    {
                        "path": file_item["path"],
                        "title": file_item["title"],
                        "type": file_item["doc"]["type"],
                        "customers": file_item["doc"]["customers"],
                        "vendors": file_item["doc"]["vendors"],
                        "commit": (file_item["doc"].get("commit") or {}).get("short_hash"),
                    }
                    for file_item in sorted(index["files"].values(), key=lambda value: value["path"])
                    if file_item["repository"] == repo and file_item["folder"] == folder_item["folder"]
                ]
                folders.append(
                    {
                        "folder": folder_item["folder"],
                        "path": folder_key,
                        "title": folder_item["title"],
                        "summary": folder_item["summary"],
                        "file_count": len(files),
                        "files": files,
                    }
                )
            repos.append(
                {
                    "repository": repo,
                    "title": item["title"],
                    "summary": item["summary"],
                    "folder_count": len(folders),
                    "file_count": sum(folder["file_count"] for folder in folders),
                    "folders": folders,
                }
            )
        return {"repositories": repos, "index": self._index_summary(index)}

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

    def _score_item(self, query: str, query_embedding: list[float], item: dict[str, Any]) -> dict[str, Any]:
        query_tokens = tokenize(query)
        item_tokens = item.get("tokens") or tokenize(item["metadata_text"])
        keyword_hits = sorted(query_tokens & item_tokens)
        keyword_score = len(keyword_hits) / max(len(query_tokens), 1)
        vector_score = max(0.0, self._cosine(query_embedding, item.get("embedding", [])))
        phrase_boost = self._phrase_boost(query, item["metadata_text"])
        total = (keyword_score * 6.0) + (vector_score * 4.0) + phrase_boost
        return {
            "score": round(total, 4),
            "keyword_score": round(keyword_score, 4),
            "vector_score": round(vector_score, 4),
            "phrase_boost": round(phrase_boost, 4),
            "keyword_hits": keyword_hits,
        }

    def _rank_items(
        self,
        query: str,
        query_embedding: list[float],
        items: list[dict[str, Any]],
        *,
        limit: int,
        minimum_score: float = 0.2,
    ) -> list[dict[str, Any]]:
        scored = []
        for item in items:
            score = self._score_item(query, query_embedding, item)
            if score["score"] >= minimum_score:
                scored.append({**item, "hybrid": score})
        scored.sort(key=lambda value: value["hybrid"]["score"], reverse=True)
        return scored[:limit]

    def _rank_repositories(
        self,
        query: str,
        query_embedding: list[float],
        index: dict[str, Any],
        repo_items: list[dict[str, Any]],
        *,
        limit: int,
        minimum_score: float = 1.0,
    ) -> list[dict[str, Any]]:
        # Repository metadata is intentionally broad, so metadata-only scores
        # cluster tightly. Blend in the strongest file-level evidence inside
        # each repository so routing follows where answers actually live.
        # A single stray token match scores well below 3.0; requiring it keeps
        # weak overlap from carrying an otherwise irrelevant repository over
        # the shortlist gate.
        minimum_file_evidence = 3.0
        allowed_repos = {item["repository"] for item in repo_items}
        best_file_score: dict[str, float] = {}
        for file_item in index["files"].values():
            repo = file_item["repository"]
            if repo not in allowed_repos:
                continue
            file_score = self._score_item(query, query_embedding, file_item)["score"]
            if file_score >= minimum_file_evidence and file_score > best_file_score.get(repo, 0.0):
                best_file_score[repo] = file_score
        scored = []
        for item in repo_items:
            own = self._score_item(query, query_embedding, item)
            evidence = best_file_score.get(item["repository"], 0.0)
            combined = round((own["score"] + evidence) / 2.0, 4)
            if combined >= minimum_score:
                hybrid = {
                    **own,
                    "score": combined,
                    "metadata_score": own["score"],
                    "best_file_score": round(evidence, 4),
                }
                scored.append({**item, "hybrid": hybrid})
        scored.sort(key=lambda value: value["hybrid"]["score"], reverse=True)
        return scored[:limit]

    def _best_snippet(self, query: str, doc: dict[str, Any]) -> tuple[str, str]:
        query_tokens = tokenize(query)
        best_heading = doc["title"]
        best_text = doc["summary"]
        best_score = -1
        for chunk in self._chunk_body(doc):
            chunk_tokens = tokenize(chunk["text"])
            score = len(query_tokens & chunk_tokens) + self._phrase_boost(query, chunk["text"])
            if score > best_score:
                best_score = score
                best_heading = chunk["heading"]
                best_text = chunk["text"]
        snippet = re.sub(r"\s+", " ", best_text).strip()
        if len(snippet) > 320:
            snippet = snippet[:317].rstrip() + "..."
        return best_heading, snippet

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

    def _folder_agent_with_openai(self, query: str, folder: dict[str, Any], selected_files: list[dict[str, Any]]) -> dict[str, Any]:
        config = model_config()
        if config["mode"] != "openai_configured" or configured_env_value("DATAMETA_DISABLE_OPENAI_SUBAGENTS") == "1":
            raise RuntimeError("OpenAI folder subagents are not enabled")
        client = self._authoring_client(config)
        docs_payload = [
            {
                "path": item["path"],
                "title": item["title"],
                "summary": item["summary"],
                "metadata": item["doc"]["metadata"],
                "full_markdown": item["doc"]["body"],
            }
            for item in selected_files
        ]
        system = (
            "You are a DataMeta folder subagent. Use only the supplied full markdown files. "
            "Return findings only when a supplied file directly supports the user query. "
            "Do not infer facts from general knowledge."
        )
        user = json.dumps(
            {
                "query": query,
                "folder": {"repository": folder["repository"], "folder": folder["folder"], "title": folder["title"]},
                "files": docs_payload,
            },
            indent=2,
            sort_keys=True,
        )
        result = client.structured_json(system=system, user=user, schema_name="datameta_folder_agent_findings", schema=FOLDER_AGENT_SCHEMA)
        valid_paths = {item["path"] for item in selected_files}
        result["findings"] = [finding for finding in result.get("findings", []) if finding.get("file_path") in valid_paths]
        result["answerable"] = bool(result["findings"]) and bool(result.get("answerable"))
        return result

    def _run_folder_agent(self, query: str, folder_item: dict[str, Any], file_items: list[dict[str, Any]], query_embedding: list[float]) -> dict[str, Any]:
        selected_files = self._rank_items(query, query_embedding, file_items, limit=3, minimum_score=1.0)
        findings = []
        agent = "local_folder_subagent"
        agent_error = None
        if selected_files:
            try:
                openai_result = self._folder_agent_with_openai(query, folder_item, selected_files)
                agent = "openai_folder_subagent"
                for finding in openai_result.get("findings", []):
                    doc_item = next(item for item in selected_files if item["path"] == finding["file_path"])
                    if not self._doc_directly_supports_query(query, doc_item["doc"]):
                        continue
                    findings.append(
                        {
                            "repository": doc_item["repository"],
                            "folder": doc_item["folder"],
                            "file_path": doc_item["path"],
                            "title": doc_item["title"],
                            "heading": finding["heading"],
                            "summary": finding["summary"],
                            "snippet": finding["snippet"],
                            "citation": self._multirepo_citation(doc_item["doc"], finding["heading"], finding["snippet"]),
                        }
                    )
            except Exception as error:
                agent_error = str(error)
                for doc_item in selected_files:
                    if not self._doc_directly_supports_query(query, doc_item["doc"]):
                        continue
                    heading, snippet = self._best_snippet(query, doc_item["doc"])
                    findings.append(
                        {
                            "repository": doc_item["repository"],
                            "folder": doc_item["folder"],
                            "file_path": doc_item["path"],
                            "title": doc_item["title"],
                            "heading": heading,
                            "summary": doc_item["summary"],
                            "snippet": snippet,
                            "citation": self._multirepo_citation(doc_item["doc"], heading, snippet),
                        }
                    )
        return {
            "repository": folder_item["repository"],
            "folder": folder_item["folder"],
            "folder_path": f"{folder_item['repository']}/{folder_item['folder']}",
            "agent": agent,
            "agent_error": agent_error,
            "answerable": bool(findings),
            "selected_files": [self._shortlist_payload(item) for item in selected_files],
            "full_content_files_read": [item["path"] for item in selected_files],
            "findings": findings,
        }

    def _shortlist_payload(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item["id"],
            "level": item["level"],
            "repository": item["repository"],
            "folder": item["folder"],
            "path": item["path"],
            "title": item["title"],
            "summary": item["summary"],
            "hybrid": item.get("hybrid"),
        }

    def _multirepo_citation(self, doc: dict[str, Any], heading: str, snippet: str) -> dict[str, Any]:
        commit = doc.get("commit") or {}
        return {
            "repository": doc["repository"],
            "folder": doc["folder"],
            "file_path": doc["path"],
            "path": doc["path"],
            "heading": heading,
            "snippet": snippet,
            "commit_hash": commit.get("hash"),
            "commit": commit.get("short_hash"),
            "title": doc["title"],
        }

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

    def _merge_hybrid_rows(self, vector_rows, chunk_rows, limit):
        vector_high = max((float(r.get("score") or 0.0) for r in vector_rows), default=0.0)
        chunk_high = max((float(r.get("score") or 0.0) for r in chunk_rows), default=0.0)

        def blank(row):
            return {"document_id": row.get("document_id"), "path": row.get("path"), "title": row.get("title"),
                    "summary": row.get("summary"), "heading": None, "text": None,
                    "vector_score": 0.0, "keyword_score": 0.0}

        docs: dict[str, dict[str, Any]] = {}
        for row in vector_rows:
            entry = docs.setdefault(row.get("document_id"), blank(row))
            entry["vector_score"] = (float(row.get("score") or 0.0) / vector_high) if vector_high else 0.0
        for row in chunk_rows:
            entry = docs.setdefault(row.get("document_id"), blank(row))
            normalized = (float(row.get("score") or 0.0) / chunk_high) if chunk_high else 0.0
            if normalized >= entry["keyword_score"]:
                entry["keyword_score"] = normalized
                entry["heading"] = row.get("heading")
                entry["text"] = row.get("text")
            entry["path"] = entry["path"] or row.get("path")
            entry["title"] = entry["title"] or row.get("title")
            entry["summary"] = entry["summary"] or row.get("summary")

        evidence: list[dict[str, Any]] = []
        for entry in docs.values():
            vs = entry["vector_score"]; ks = entry["keyword_score"]
            score = round(vs * 0.6 + ks * 0.4, 4)
            body = entry["text"] or entry["summary"] or ""
            text = re.sub(r"\s+", " ", body).strip()
            snippet = (text[:317].rstrip() + "...") if len(text) > 320 else text
            heading = entry["heading"] or entry["title"] or ""
            path = entry["path"] or ""
            parts = path.split("/")
            evidence.append({
                "path": path, "heading": heading, "snippet": snippet,
                "title": entry["title"] or path, "summary": entry["summary"] or "",
                "score": score, "vector_score": round(vs, 4), "keyword_score": round(ks, 4),
                "citation": {"repository": parts[0] if parts else "",
                             "folder": "/".join(parts[1:-1]) if len(parts) > 2 else "",
                             "file_path": path, "path": path, "heading": heading,
                             "snippet": snippet, "document_id": entry["document_id"],
                             "title": entry["title"] or path}})
        evidence.sort(key=lambda item: item["score"], reverse=True)
        return evidence[:limit]

    def _neo4j_hybrid_retrieve(self, user, query, query_embedding, limit=8):
        if not self._neo4j_status()["configured"]:
            return None
        if len(query_embedding) != self._neo4j_vector_dimensions():
            return None
        is_admin = user.is_admin
        teams = list(user.read_teams)
        vector_rows = self._neo4j_read(
            "CALL db.index.vector.queryNodes('datameta_document_embedding', $k, $vec) YIELD node, score "
            "MATCH (node)-[:OWNED_BY]->(t:Team) WHERE $admin OR t.name IN $teams "
            "RETURN node.id AS document_id, node.path AS path, node.title AS title, node.summary AS summary, "
            "t.name AS team, score AS score",
            {"k": limit * 2, "vec": query_embedding, "admin": is_admin, "teams": teams}) or []
        chunk_rows = self._neo4j_read(
            "CALL db.index.fulltext.queryNodes('datameta_chunk_fulltext', $q) YIELD node, score "
            "MATCH (d:Document)-[:HAS_CHUNK]->(node) MATCH (d)-[:OWNED_BY]->(t:Team) "
            "WHERE $admin OR t.name IN $teams "
            "RETURN d.id AS document_id, d.path AS path, d.title AS title, d.summary AS summary, "
            "node.heading AS heading, node.text AS text, t.name AS team, score AS score",
            {"q": query, "admin": is_admin, "teams": teams}) or []
        if not is_admin:
            allowed = set(teams)
            vector_rows = [r for r in vector_rows if r.get("team") in allowed]
            chunk_rows = [r for r in chunk_rows if r.get("team") in allowed]
        # Return a wider candidate pool; the caller filters and re-ranks it
        # against the local index before truncating to `limit`.
        return {
            "source": "neo4j_hybrid",
            "limit": limit,
            "evidence": self._merge_hybrid_rows(vector_rows, chunk_rows, limit * 2),
        }

    def _blend_neo4j_evidence_with_local(
        self,
        query: str,
        query_embedding: list[float],
        evidence: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        # Neo4j metadata-embedding scores cluster tightly, so blend in the
        # local hybrid score (keyword + entity phrase signal) of the mirrored
        # document to rank decisive evidence above loosely related documents.
        index = self._multirepo_index
        local_scores: dict[str, float] = {}
        for item in evidence:
            file_item = index["files"].get(item.get("path", "")) if index else None
            if file_item:
                local_scores[item["path"]] = self._score_item(query, query_embedding, file_item)["score"]
        local_best = max(local_scores.values(), default=0.0)
        if local_best > 0:
            for item in evidence:
                local_score = local_scores.get(item.get("path", ""), 0.0)
                item["score"] = round(0.5 * item["score"] + 0.5 * (local_score / local_best), 4)
            evidence = sorted(evidence, key=lambda value: value["score"], reverse=True)
        return evidence[:limit]

    def _evidence_supports_query(self, query: str, item: dict[str, Any]) -> bool:
        index = self._multirepo_index
        file_item = index["files"].get(item.get("path", "")) if index else None
        if not file_item:
            # Unknown to the local mirror (e.g. Neo4j-only document): keep it,
            # the local corpus cannot rule on it.
            return True
        return self._doc_directly_supports_query(query, file_item["doc"])

    def _local_evidence_from_findings(self, findings, shortlisted_files):
        hybrid_by_path = {item["path"]: (item.get("hybrid") or {}) for item in shortlisted_files}
        evidence = []
        for finding in findings:
            hybrid = hybrid_by_path.get(finding["file_path"], {})
            evidence.append({"path": finding["file_path"], "heading": finding["heading"],
                             "snippet": finding["snippet"], "title": finding["title"], "summary": finding["summary"],
                             "score": hybrid.get("score", 0.0), "vector_score": hybrid.get("vector_score", 0.0),
                             "keyword_score": hybrid.get("keyword_score", 0.0), "citation": finding["citation"]})
        return evidence

    def datameta_multirepo_query(self, user_id: str | None, query: str, include_trace: bool = True) -> dict[str, Any]:
        return self.multirepo_query(user_id, query, include_trace)

    def multirepo_query(self, user_id: str | None, query: str, include_trace: bool = True) -> dict[str, Any]:
        user = self.get_user(user_id)
        self.index_repos()
        assert self._multirepo_index is not None
        index = self._multirepo_index
        query_embeddings, query_embedding_status = self._embed_texts([query])
        query_embedding = query_embeddings[0]
        repo_items = [item for item in index["repositories"].values() if user.can_read_team(item["doc"]["team"])]
        shortlisted_repositories = self._rank_repositories(query, query_embedding, index, repo_items, limit=4, minimum_score=1.0)
        folder_candidates: list[dict[str, Any]] = []
        for repo_item in shortlisted_repositories:
            folders = [
                item
                for item in index["folders"].values()
                if item["repository"] == repo_item["repository"] and user.can_read_team(item["doc"]["team"])
            ]
            folder_candidates.extend(self._rank_items(query, query_embedding, folders, limit=3, minimum_score=1.0))
        file_scores_by_folder: dict[str, list[dict[str, Any]]] = {}
        for folder_item in folder_candidates:
            folder_key = f"{folder_item['repository']}/{folder_item['folder']}"
            file_scores_by_folder[folder_key] = [
                item
                for item in index["files"].values()
                if item["repository"] == folder_item["repository"]
                and item["folder"] == folder_item["folder"]
                and user.can_read_team(item["doc"]["team"])
            ]
        folder_findings = []
        folder_scores = {
            f"{folder_item['repository']}/{folder_item['folder']}": folder_item.get("hybrid", {}).get("score", 0)
            for folder_item in folder_candidates
        }
        if folder_candidates:
            with ThreadPoolExecutor(max_workers=min(8, len(folder_candidates))) as executor:
                futures = {
                    executor.submit(
                        self._run_folder_agent,
                        query,
                        folder_item,
                        file_scores_by_folder[f"{folder_item['repository']}/{folder_item['folder']}"],
                        query_embedding,
                    ): folder_item
                    for folder_item in folder_candidates
                }
                for future in as_completed(futures):
                    folder_findings.append(future.result())
        folder_findings.sort(key=lambda value: folder_scores.get(value["folder_path"], 0), reverse=True)
        findings = [finding for folder in folder_findings for finding in folder["findings"]]
        shortlisted_files = [
            file_payload
            for folder in folder_findings
            for file_payload in folder["selected_files"]
        ]
        neo4j_retrieval = self._neo4j_hybrid_retrieve(user, query, query_embedding)
        if neo4j_retrieval and neo4j_retrieval.get("evidence"):
            # Apply the same no-guessing entity filter used for local findings,
            # so e.g. a Customer C document is not cited for a Customer A query.
            supported = [
                item
                for item in neo4j_retrieval["evidence"]
                if self._evidence_supports_query(query, item)
            ]
            neo4j_retrieval["evidence"] = self._blend_neo4j_evidence_with_local(
                query, query_embedding, supported, neo4j_retrieval.get("limit", 8)
            )
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
        trace = None
        if include_trace:
            trace = {
                "namespace_order": ["repository", "folder", "file"],
                "repo_hybrid_scores": [self._shortlist_payload(item) for item in shortlisted_repositories],
                "folder_hybrid_scores": [self._shortlist_payload(item) for item in folder_candidates],
                "file_hybrid_scores": shortlisted_files,
                "folder_subagents_spawned": [folder["folder_path"] for folder in folder_findings],
                "full_markdown_read_after_file_metadata_selection": {
                    folder["folder_path"]: folder["full_content_files_read"] for folder in folder_findings
                },
                "embedding": index["embedding"],
                "query_embedding": query_embedding_status,
                "neo4j": index["neo4j"],
                "retrieval_source": retrieval_source,
                "no_guessing_policy": "Return not answerable when selected files do not directly support the answer.",
            }
        return {
            "query": query,
            "user": self._user_payload(user),
            "answerable": answerable,
            "answer": answer_text,
            "shortlisted_repositories": [self._shortlist_payload(item) for item in shortlisted_repositories],
            "shortlisted_folders": [self._shortlist_payload(item) for item in folder_candidates],
            "shortlisted_files": shortlisted_files,
            "folder_subagent_findings": folder_findings,
            "citations": citations,
            "retrieval_source": retrieval_source,
            "retrieval": retrieval,
            "trace": trace,
        }

    def retrieve(self, user_id: str | None, query: str, include_trace: bool = False) -> dict[str, Any]:
        result = self.multirepo_query(user_id, query, include_trace)
        packets = [
            {
                "score": file_item.get("hybrid", {}).get("score"),
                "id": file_item["id"],
                "path": file_item["path"],
                "team": file_item["repository"],
                "type": "file",
                "entity": file_item["title"],
                "scope": file_item["folder"],
                "title": file_item["title"],
                "summary": file_item["summary"],
                "snippet": next((citation["snippet"] for citation in result["citations"] if citation["file_path"] == file_item["path"]), ""),
                "citation": next((citation for citation in result["citations"] if citation["file_path"] == file_item["path"]), None),
            }
            for file_item in result["shortlisted_files"]
        ]
        return {"query": query, "user": result["user"], "packets": packets, "trace": result["trace"]}

    def answer(self, user_id: str | None, question: str, include_trace: bool = False) -> dict[str, Any]:
        result = self.multirepo_query(user_id, question, include_trace)
        result["question"] = question
        return result

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
        # Calculation intent: if the question names a metric we have definitions
        # for, surface them. When more than one definition is visible, pause and
        # ask which one to use instead of guessing.
        prepared = self.prepare_calculation(question, user_id)
        if prepared["definitions"]:
            if prepared["requires_choice"]:
                return {"intent": "choose_definition", "action": "datameta_choose_definition", **prepared}
            definition = prepared["definitions"][0]
            table = definition["accessible_tables"][0]["table"] if definition["accessible_tables"] else None
            if table:
                calculation = self.run_calculation(user_id, definition["id"], table)
                return {
                    "intent": "calculation",
                    "action": "datameta_calculation",
                    "prepare": prepared,
                    "calculation": calculation,
                }
            return {"intent": "choose_definition", "action": "datameta_choose_definition", **prepared}
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
                "example": "incident_events table, evt-002 Vendor X recovery evidence is missing from the Customer A SLA review.",
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
        tokens = tokenize(text)
        if {"availability", "sla", "slo", "uptime"} & tokens and user.can_access_table("sla_availability"):
            return "sla_availability"
        if {"vendor", "rca"} & tokens and user.can_access_table("vendor_events"):
            return "vendor_events"
        if {"incident", "outage", "timeline", "event", "events"} & tokens and user.can_access_table("incident_events"):
            return "incident_events"
        if {"arr", "mrr", "acv", "subscription", "subscriptions", "revenue"} & tokens and user.can_access_table("arr_subscriptions"):
            return "arr_subscriptions"
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
        commit_hash = self._git_commit(f"DataMeta: resolve outlier {flag_id}", user.name, f"{user.id}@generic-enterprise.local")
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

    def history(self, path: str | None = None, user_id: str | None = None) -> dict[str, Any]:
        self.ensure_ready()
        user = self.get_user(user_id)
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
                for doc in self.visible_documents(user)
            ],
            "flags": self.list_flags(),
            "pipeline_runs": self.list_pipeline_runs(),
        }

    def _markdown_search_payload(
        self,
        doc: dict[str, Any],
        query: str,
        score: float,
        keyword_hits: list[str],
        *,
        backend: str,
    ) -> dict[str, Any]:
        heading, snippet = self._best_snippet(query or doc["title"], doc)
        return {
            "score": score,
            "keyword_hits": keyword_hits,
            "backend": backend,
            "id": doc["id"],
            "path": doc["path"],
            "repository": doc["repository"],
            "folder": doc["folder"],
            "metadata_level": doc["metadata_level"],
            "team": doc["team"],
            "type": doc["type"],
            "title": doc["title"],
            "summary": doc["summary"],
            "customers": doc["customers"],
            "vendors": doc["vendors"],
            "heading": heading,
            "snippet": snippet,
            "citation": self._citation(doc),
        }

    def search_markdown_files(self, user_id: str | None, query: str = "", limit: int = 50) -> dict[str, Any]:
        user = self.get_user(user_id)
        safe_limit = max(1, min(int(limit or 50), 200))
        query_tokens = tokenize(query)
        results = []
        for doc in self.visible_documents(user):
            haystack = " ".join([doc["path"], doc["title"], doc["summary"], doc["entity"], doc["scope"], doc["body"]])
            doc_tokens = tokenize(haystack)
            keyword_hits = sorted(query_tokens & doc_tokens)
            if query_tokens:
                score = len(keyword_hits)
                if query.lower() in haystack.lower():
                    score += 5
                if score <= 0:
                    continue
            else:
                score = 0
            results.append(self._markdown_search_payload(doc, query, score, keyword_hits, backend="local"))
        results.sort(key=lambda item: (-item["score"], item["path"]))
        return {
            "query": query,
            "user": self._user_payload(user),
            "files": results[:safe_limit],
            "total": len(results),
            "backend": "local",
        }

    def read_markdown_file(self, user_id: str | None, path: str) -> dict[str, Any]:
        user = self.get_user(user_id)
        normalized = path.strip().lstrip("/")
        if not normalized or ".." in Path(normalized).parts or not normalized.endswith(".md"):
            raise ValueError("Markdown path must be a repository-relative .md file")
        for doc in self.visible_documents(user):
            if doc["path"] == normalized:
                return {
                    "id": doc["id"],
                    "path": doc["path"],
                    "repository": doc["repository"],
                    "folder": doc["folder"],
                    "metadata_level": doc["metadata_level"],
                    "team": doc["team"],
                    "type": doc["type"],
                    "title": doc["title"],
                    "summary": doc["summary"],
                    "metadata": doc["metadata"],
                    "body": doc["body"],
                    "markdown": doc["text"],
                    "citation": self._citation(doc),
                    "user": self._user_payload(user),
                }
        raise PermissionError(f"{user.name} cannot access markdown file {normalized}")

    def bootstrap(self, user_id: str | None = None) -> dict[str, Any]:
        user = self.get_user(user_id)
        return {
            "project": "DataMeta",
            "company": "Generic Enterprise",
            "models": model_config(),
            "user": self._user_payload(user),
            "users": self.users(),
            "schema": self.warehouse_schema(user),
            "history": self.history(user_id=user.id),
            "inventory": self.repo_inventory(),
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
            "datameta_index_repos",
            "datameta_repo_inventory",
            "datameta_multirepo_query",
            "datameta_search_markdown",
            "datameta_read_markdown_file",
            "datameta_author_proposal",
            "datameta_validate_proposal",
            "datameta_commit_proposal",
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
