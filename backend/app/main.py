from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .datameta import DataMetaService


service = DataMetaService()

app = FastAPI(title="DataMeta", version="0.1.0")


def allowed_origins() -> list[str]:
    configured = os.environ.get("DATAMETA_ALLOWED_ORIGINS")
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return ["http://localhost:3000", "http://127.0.0.1:3000"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_origin_regex=r"https://.*\.vercel\.app" if os.environ.get("VERCEL") else None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    service.ensure_ready()


class QuestionRequest(BaseModel):
    question: str
    user_id: str | None = None
    include_trace: bool = False


class AuthorRequest(BaseModel):
    natural_language: str
    user_id: str | None = None
    target_team: str | None = None


class ProposalRequest(BaseModel):
    proposal_id: str
    user_id: str | None = None


class CommitProposalRequest(ProposalRequest):
    confirm_overwrite: bool = False


class CalculationRunRequest(BaseModel):
    user_id: str | None = None
    definition_id: str
    table: str


class FlagOutlierRequest(BaseModel):
    user_id: str | None = None
    table_name: str = "orders"
    subject: str
    description: str
    owner_team: str = "data-ownership"


class ResolveOutlierRequest(BaseModel):
    user_id: str | None = None
    flag_id: str
    resolution: str


class PipelineRunRequest(BaseModel):
    user_id: str | None = None
    runbook_id: str = "gmv-category-ranker"
    variant: str | None = None


def user_from_header(header_user: str | None, body_user: str | None = None) -> str | None:
    return body_user or header_user


def handle_error(error: Exception) -> HTTPException:
    if isinstance(error, PermissionError):
        return HTTPException(status_code=403, detail=str(error))
    if isinstance(error, ValueError):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=500, detail=str(error))


@app.get("/api/bootstrap")
def bootstrap(x_user_id: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        return service.bootstrap(x_user_id)
    except Exception as error:
        raise handle_error(error)


@app.post("/api/retrieve")
def retrieve(request: QuestionRequest, x_user_id: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        return service.retrieve(user_from_header(x_user_id, request.user_id), request.question, request.include_trace)
    except Exception as error:
        raise handle_error(error)


@app.post("/api/answer")
def answer(request: QuestionRequest, x_user_id: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        return service.answer(user_from_header(x_user_id, request.user_id), request.question, request.include_trace)
    except Exception as error:
        raise handle_error(error)


@app.post("/api/ask")
def ask(request: QuestionRequest, x_user_id: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        return service.ask(user_from_header(x_user_id, request.user_id), request.question, request.include_trace)
    except Exception as error:
        raise handle_error(error)


@app.post("/api/author/proposal")
def author_proposal(request: AuthorRequest, x_user_id: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        return service.create_author_proposal(
            user_from_header(x_user_id, request.user_id),
            request.natural_language,
            request.target_team,
        )
    except Exception as error:
        raise handle_error(error)


@app.post("/api/author/validate")
def validate_proposal(request: ProposalRequest, x_user_id: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        return service.validate_proposal(request.proposal_id, user_from_header(x_user_id, request.user_id))
    except Exception as error:
        raise handle_error(error)


@app.post("/api/author/commit")
def commit_proposal(request: CommitProposalRequest, x_user_id: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        return service.commit_proposal(
            user_from_header(x_user_id, request.user_id),
            request.proposal_id,
            request.confirm_overwrite,
        )
    except Exception as error:
        raise handle_error(error)


@app.post("/api/calculation/prepare")
def prepare_calculation(request: QuestionRequest, x_user_id: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        return service.prepare_calculation(request.question, user_from_header(x_user_id, request.user_id))
    except Exception as error:
        raise handle_error(error)


@app.post("/api/calculation/run")
def run_calculation(request: CalculationRunRequest, x_user_id: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        return service.run_calculation(user_from_header(x_user_id, request.user_id), request.definition_id, request.table)
    except Exception as error:
        raise handle_error(error)


@app.post("/api/outliers/flag")
def flag_outlier(request: FlagOutlierRequest, x_user_id: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        return service.flag_outlier(
            user_from_header(x_user_id, request.user_id),
            request.table_name,
            request.subject,
            request.description,
            request.owner_team,
        )
    except Exception as error:
        raise handle_error(error)


@app.post("/api/outliers/resolve")
def resolve_flag(request: ResolveOutlierRequest, x_user_id: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        return service.resolve_flag(user_from_header(x_user_id, request.user_id), request.flag_id, request.resolution)
    except Exception as error:
        raise handle_error(error)


@app.post("/api/pipeline/run")
def run_pipeline(request: PipelineRunRequest, x_user_id: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        return service.run_pipeline(user_from_header(x_user_id, request.user_id), request.runbook_id, request.variant)
    except Exception as error:
        raise handle_error(error)


@app.get("/api/history")
def history(path: str | None = None) -> dict[str, Any]:
    try:
        return service.history(path)
    except Exception as error:
        raise handle_error(error)


def mcp_tools() -> list[dict[str, Any]]:
    user_property = {"type": "string", "description": "Seeded DataMeta user id. Defaults to junior.analyst."}
    return [
        {
            "name": "datameta_ask",
            "title": "Ask DataMeta",
            "description": (
                "Natural-language DataMeta entrypoint for Shoppy definitions, runbooks, data-quality concerns, "
                "and calculation questions. Use this when the user asks in plain English."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "user_id": user_property,
                    "include_trace": {"type": "boolean"},
                },
                "required": ["question"],
            },
            "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
        },
        {
            "name": "datameta_retrieve",
            "title": "Retrieve DataMeta Knowledge",
            "description": "Retrieve Shoppy GraphRAG context with citations and RBAC filtering.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "user_id": user_property,
                    "include_trace": {"type": "boolean"},
                },
                "required": ["question"],
            },
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "datameta_answer",
            "title": "Answer With DataMeta",
            "description": "Answer a Shoppy question using committed DataMeta knowledge and citations.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "user_id": user_property,
                    "include_trace": {"type": "boolean"},
                },
                "required": ["question"],
            },
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "datameta_author_proposal",
            "title": "Draft Knowledge Proposal",
            "description": "Draft markdown knowledge from natural language. Does not commit.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "natural_language": {"type": "string"},
                    "target_team": {"type": "string"},
                    "user_id": user_property,
                },
                "required": ["natural_language"],
            },
            "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
        },
        {
            "name": "datameta_validate_proposal",
            "title": "Validate Knowledge Proposal",
            "description": "Validate RBAC, schema samples, and same-entity/same-scope conflicts.",
            "inputSchema": {
                "type": "object",
                "properties": {"proposal_id": {"type": "string"}, "user_id": user_property},
                "required": ["proposal_id"],
            },
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "datameta_commit_proposal",
            "title": "Commit Knowledge Proposal",
            "description": "Commit a validated proposal to Git after explicit confirmation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "string"},
                    "confirm_overwrite": {"type": "boolean"},
                    "user_id": user_property,
                },
                "required": ["proposal_id"],
            },
            "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
        },
        {
            "name": "datameta_prepare_calculation",
            "title": "Prepare Calculation",
            "description": "Find latest visible definitions, explain options, and list accessible tables.",
            "inputSchema": {
                "type": "object",
                "properties": {"question": {"type": "string"}, "user_id": user_property},
                "required": ["question"],
            },
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "datameta_run_calculation",
            "title": "Run Calculation",
            "description": "Run a selected committed definition against an accessible table using controlled read-only SQL.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "definition_id": {"type": "string"},
                    "table": {"type": "string"},
                    "user_id": user_property,
                },
                "required": ["definition_id", "table"],
            },
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "datameta_flag_outlier",
            "title": "Flag Outlier",
            "description": "Flag suspected bad data for owner-team review.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string"},
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                    "owner_team": {"type": "string"},
                    "user_id": user_property,
                },
                "required": ["table_name", "subject", "description"],
            },
            "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
        },
        {
            "name": "datameta_resolve_flag",
            "title": "Resolve Outlier Flag",
            "description": "Resolve a data-quality flag and commit the official note to Git.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "flag_id": {"type": "string"},
                    "resolution": {"type": "string"},
                    "user_id": user_property,
                },
                "required": ["flag_id", "resolution"],
            },
            "annotations": {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": False},
        },
        {
            "name": "datameta_run_pipeline",
            "title": "Run DataMeta Pipeline",
            "description": "Execute an approved versioned runbook and return chart/table outputs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "runbook_id": {"type": "string"},
                    "variant": {"type": "string"},
                    "user_id": user_property,
                },
            },
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
        {
            "name": "datameta_history",
            "title": "Show DataMeta History",
            "description": "Show Git commits, documents, flags, and pipeline run history.",
            "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        },
    ]


def mcp_text_result(payload: Any) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": payload if isinstance(payload, str) else json_dumps(payload)}],
        "structuredContent": payload if isinstance(payload, dict) else {"result": payload},
        "isError": False,
    }


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def dispatch_tool(name: str, arguments: dict[str, Any]) -> Any:
    if name == "datameta_ask":
        return service.ask(arguments.get("user_id"), arguments["question"], bool(arguments.get("include_trace", False)))
    if name == "datameta_retrieve":
        return service.retrieve(arguments.get("user_id"), arguments["question"], bool(arguments.get("include_trace", False)))
    if name == "datameta_answer":
        return service.answer(arguments.get("user_id"), arguments["question"], bool(arguments.get("include_trace", False)))
    if name == "datameta_author_proposal":
        return service.create_author_proposal(arguments.get("user_id"), arguments["natural_language"], arguments.get("target_team"))
    if name == "datameta_validate_proposal":
        return service.validate_proposal(arguments["proposal_id"], arguments.get("user_id"))
    if name == "datameta_commit_proposal":
        return service.commit_proposal(arguments.get("user_id"), arguments["proposal_id"], bool(arguments.get("confirm_overwrite", False)))
    if name == "datameta_prepare_calculation":
        return service.prepare_calculation(arguments["question"], arguments.get("user_id"))
    if name == "datameta_run_calculation":
        return service.run_calculation(arguments.get("user_id"), arguments["definition_id"], arguments["table"])
    if name == "datameta_flag_outlier":
        return service.flag_outlier(
            arguments.get("user_id"),
            arguments["table_name"],
            arguments["subject"],
            arguments["description"],
            arguments.get("owner_team", "data-ownership"),
        )
    if name == "datameta_resolve_flag":
        return service.resolve_flag(arguments.get("user_id"), arguments["flag_id"], arguments["resolution"])
    if name == "datameta_run_pipeline":
        return service.run_pipeline(arguments.get("user_id"), arguments.get("runbook_id", "gmv-category-ranker"), arguments.get("variant"))
    if name == "datameta_history":
        return service.history(arguments.get("path"))
    raise ValueError(f"Unknown DataMeta tool: {name}")


@app.post("/mcp", response_model=None)
async def mcp(request: Request) -> dict[str, Any] | Response:
    expected_token = os.environ.get("DATAMETA_MCP_TOKEN")
    if expected_token:
        authorization = request.headers.get("authorization", "")
        if authorization != f"Bearer {expected_token}":
            raise HTTPException(status_code=401, detail="Invalid DataMeta MCP bearer token")
    message = await request.json()
    method = message.get("method")
    request_id = message.get("id")
    try:
        if method == "initialize":
            result = {
                "protocolVersion": message.get("params", {}).get("protocolVersion", "2025-06-18"),
                "serverInfo": {"name": "datameta", "title": "DataMeta", "version": "0.1.0"},
                "capabilities": {"tools": {"listChanged": False}},
                "instructions": (
                    "Use DataMeta for Shoppy company definitions, runbooks, data-quality flags, and calculations. "
                    "For plain-language user questions, call datameta_ask first. "
                    "Call datameta_prepare_calculation before calculating ARR so the analyst can choose Finance or Renewals. "
                    "Cite returned paths and commit hashes."
                ),
            }
        elif method == "tools/list":
            result = {"tools": mcp_tools()}
        elif method == "tools/call":
            params = message.get("params", {})
            payload = dispatch_tool(params["name"], params.get("arguments") or {})
            result = mcp_text_result(payload)
        elif method in {"notifications/initialized", "initialized"}:
            return Response(status_code=202)
        else:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"Unknown method {method}"}}
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except Exception as error:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": str(error)}],
                "isError": True,
            },
        }
