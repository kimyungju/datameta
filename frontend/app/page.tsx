"use client";

import {
  AlertTriangle,
  BarChart3,
  BookOpenText,
  Check,
  ChevronRight,
  Database,
  FileClock,
  GitCommitHorizontal,
  History,
  KeyRound,
  LogIn,
  Loader2,
  MessageSquareText,
  Play,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  SquarePen
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

type User = {
  id: string;
  name: string;
  roles: string[];
  read_teams: string[];
  write_teams: string[];
  tables: string[];
};

type Citation = {
  path: string;
  commit?: string;
  commit_hash?: string;
  author?: string;
  authored_at?: string;
  title?: string;
};

type Definition = {
  id: string;
  title: string;
  team: string;
  scope: string;
  summary: string;
  entails: string[];
  required_columns: string[];
  accessible_tables: { table: string; columns: string[] }[];
  citation: Citation;
};

type PrepareResult = {
  message: string;
  requires_choice: boolean;
  definitions: Definition[];
};

type CalculationResult = {
  ok: boolean;
  definition: Definition;
  table: string;
  sql?: string;
  result?: { label: string; value: number };
  citation?: Citation;
  blocked?: boolean;
  missing_columns?: string[];
  message?: string;
};

type Bootstrap = {
  project: string;
  company: string;
  models: {
    mode: string;
    api_key_configured: boolean;
    reasoning?: string | null;
    embedding?: string | null;
  };
  user: User;
  users: User[];
  schema: { table: string; columns: string[] }[];
  history: HistoryPayload;
};

type HistoryPayload = {
  documents: Array<{
    id: string;
    path: string;
    team: string;
    type: string;
    entity: string;
    scope: string;
    title: string;
    summary: string;
    citation: Citation;
  }>;
  commits: Array<{
    hash: string;
    short_hash: string;
    author: string;
    authored_at: string;
    subject: string;
    paths: string[];
  }>;
  flags: OutlierFlag[];
  pipeline_runs: PipelineRun[];
};

type OutlierFlag = {
  id: string;
  created_at: string;
  created_by: string;
  owner_team: string;
  table_name: string;
  subject: string;
  description: string;
  status: string;
  resolved_at?: string;
  resolved_by?: string;
  resolution?: string;
};

type PipelineRun = {
  id: string;
  created_at: string;
  created_by: string;
  runbook_id: string;
  output: PipelineOutput;
};

type PipelineOutput = {
  table: Array<{ category: string; net_gmv: number; order_count: number }>;
  chart: Array<{ label: string; value: number; color: string }>;
  sql: string;
  citation: Citation;
  runbook: { title: string; team: string; scope: string };
};

type ProposalResult = {
  proposal_id: string;
  proposal: Record<string, unknown>;
  markdown: string;
  validation: ValidationResult;
  model?: string | null;
};

type ValidationResult = {
  ok: boolean;
  blocked: boolean;
  needs_confirmation: boolean;
  checks: Array<{
    name: string;
    ok: boolean;
    detail: string;
    conflicts?: unknown[];
    matching_tables?: unknown[];
  }>;
  samples: Record<string, Record<string, unknown[]>>;
};

type ApiError = { detail?: string };

type Tab = "calculate" | "query" | "author" | "pipeline" | "outliers" | "history";

function money(value: number | undefined) {
  if (typeof value !== "number") return "-";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(value);
}

function dateLabel(value?: string) {
  if (!value) return "";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

function modelStatus(models?: Bootstrap["models"]) {
  if (!models) return "Local deterministic";
  const configuredModels = [models.reasoning, models.embedding].filter(Boolean).join(" · ");
  if (models.mode === "openai_missing_model") return "OpenAI key configured · model missing";
  const mode = models.api_key_configured ? "OpenAI configured" : "Local deterministic";
  return configuredModels ? `${configuredModels} · ${mode}` : mode;
}

export default function Home() {
  const [tab, setTab] = useState<Tab>("calculate");
  const [userId, setUserId] = useState("");
  const [loginUserId, setLoginUserId] = useState("junior.analyst");
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  const [calcQuestion, setCalcQuestion] = useState("How to calculate ARR?");
  const [prepare, setPrepare] = useState<PrepareResult | null>(null);
  const [selectedDefinition, setSelectedDefinition] = useState("");
  const [selectedTable, setSelectedTable] = useState("");
  const [calculation, setCalculation] = useState<CalculationResult | null>(null);

  const [query, setQuery] = useState("How do we calculate ARR for a board update?");
  const [answer, setAnswer] = useState<Record<string, unknown> | null>(null);
  const [includeTrace, setIncludeTrace] = useState(false);

  const [authorText, setAuthorText] = useState("Finance definition of ARR for board reporting is active MRR times 12.");
  const [targetTeam, setTargetTeam] = useState("finance");
  const [proposal, setProposal] = useState<ProposalResult | null>(null);
  const [commitResult, setCommitResult] = useState<Record<string, unknown> | null>(null);

  const [pipeline, setPipeline] = useState<PipelineOutput | null>(null);
  const [flagSubject, setFlagSubject] = useState("May 4 GMV spike");
  const [flagDescription, setFlagDescription] = useState("GMV is more than 10x the previous three days. Ops suspects a duplicated marketplace feed order.");
  const [flagResult, setFlagResult] = useState<Record<string, unknown> | null>(null);
  const [resolutionText, setResolutionText] = useState("The marketplace feed duplicated a single order. Exclude ord_005 until the data fix lands.");

  const activeUser = useMemo(() => bootstrap?.users.find((user) => user.id === userId), [bootstrap, userId]);
  const loginUser = useMemo(() => bootstrap?.users.find((user) => user.id === loginUserId), [bootstrap, loginUserId]);

  async function api<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        "x-user-id": userId,
        ...(init?.headers ?? {})
      }
    });
    if (!response.ok) {
      const payload = (await response.json().catch(() => ({}))) as ApiError;
      throw new Error(payload.detail ?? `Request failed: ${response.status}`);
    }
    return response.json() as Promise<T>;
  }

  async function refresh(nextUser = userId) {
    setError("");
    const response = await fetch(`${API_BASE}/api/bootstrap`, {
      headers: { "x-user-id": nextUser || "junior.analyst" }
    });
    if (!response.ok) throw new Error("Backend is not available on port 8000");
    const payload = (await response.json()) as Bootstrap;
    setBootstrap(payload);
  }

  async function runAction<T>(action: () => Promise<T>, success = "") {
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const result = await action();
      if (success) setNotice(success);
      return result;
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      return null;
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh("junior.analyst").catch((caught) => setError(caught instanceof Error ? caught.message : String(caught)));
  }, []);

  useEffect(() => {
    if (!userId) return;
    setPrepare(null);
    setCalculation(null);
    setAnswer(null);
    refresh(userId).catch((caught) => setError(caught instanceof Error ? caught.message : String(caught)));
  }, [userId]);

  function login() {
    setNotice("");
    setError("");
    setUserId(loginUserId);
  }

  function switchUser() {
    setLoginUserId(userId || loginUserId);
    setUserId("");
    setPrepare(null);
    setCalculation(null);
    setAnswer(null);
    setProposal(null);
    setCommitResult(null);
    setPipeline(null);
    setFlagResult(null);
    setNotice("");
  }

  async function prepareCalculation() {
    const result = await runAction(
      () =>
        api<PrepareResult>("/api/calculation/prepare", {
          method: "POST",
          body: JSON.stringify({ question: calcQuestion })
        }),
      "Definitions loaded"
    );
    if (result) {
      setPrepare(result);
      const first = result.definitions[0];
      setSelectedDefinition(first?.id ?? "");
      setSelectedTable(first?.accessible_tables[0]?.table ?? "");
      setCalculation(null);
    }
  }

  async function runCalculation() {
    const result = await runAction(
      () =>
        api<CalculationResult>("/api/calculation/run", {
          method: "POST",
          body: JSON.stringify({ definition_id: selectedDefinition, table: selectedTable })
        }),
      "Calculation complete"
    );
    if (result) setCalculation(result);
  }

  async function askQuestion() {
    const result = await runAction(() =>
      api<Record<string, unknown>>("/api/answer", {
        method: "POST",
        body: JSON.stringify({ question: query, include_trace: includeTrace })
      })
    );
    if (result) setAnswer(result);
  }

  async function draftProposal() {
    const result = await runAction(
      () =>
        api<ProposalResult>("/api/author/proposal", {
          method: "POST",
          body: JSON.stringify({ natural_language: authorText, target_team: targetTeam })
        }),
      "Proposal drafted"
    );
    if (result) {
      setProposal(result);
      setCommitResult(null);
    }
  }

  async function commitProposal(confirmOverwrite: boolean) {
    if (!proposal) return;
    const result = await runAction(
      () =>
        api<Record<string, unknown>>("/api/author/commit", {
          method: "POST",
          body: JSON.stringify({ proposal_id: proposal.proposal_id, confirm_overwrite: confirmOverwrite })
        }),
      confirmOverwrite ? "Confirmed and committed" : "Commit checked"
    );
    if (result) {
      setCommitResult(result);
      refresh().catch(() => undefined);
    }
  }

  async function runPipeline() {
    const result = await runAction(
      () =>
        api<{ output: PipelineOutput }>("/api/pipeline/run", {
          method: "POST",
          body: JSON.stringify({ runbook_id: "gmv-category-ranker" })
        }),
      "Pipeline run saved"
    );
    if (result) {
      setPipeline(result.output);
      refresh().catch(() => undefined);
    }
  }

  async function flagOutlier() {
    const result = await runAction(
      () =>
        api<Record<string, unknown>>("/api/outliers/flag", {
          method: "POST",
          body: JSON.stringify({
            table_name: "orders",
            subject: flagSubject,
            description: flagDescription,
            owner_team: "data-ownership"
          })
        }),
      "Flag sent to data ownership"
    );
    if (result) {
      setFlagResult(result);
      refresh().catch(() => undefined);
    }
  }

  async function resolveFlag(flagId: string) {
    const result = await runAction(
      () =>
        api<Record<string, unknown>>("/api/outliers/resolve", {
          method: "POST",
          body: JSON.stringify({ flag_id: flagId, resolution: resolutionText })
        }),
      "Flag resolved and committed"
    );
    if (result) {
      setFlagResult(result);
      refresh().catch(() => undefined);
    }
  }

  const currentDefinition = prepare?.definitions.find((definition) => definition.id === selectedDefinition);
  const tableOptions = currentDefinition?.accessible_tables ?? [];
  const maxChart = pipeline?.chart.reduce((max, item) => Math.max(max, item.value), 0) ?? 1;

  if (!userId) {
    return (
      <main className="login-shell">
        <section className="login-panel">
          <div className="brand login-brand">
            <div className="mark">DM</div>
            <div>
              <h1>DataMeta</h1>
              <p>Shoppy</p>
            </div>
          </div>

          <div className="login-copy">
            <p className="eyebrow">{modelStatus(bootstrap?.models)} · RBAC demo</p>
            <h2>Choose your user</h2>
          </div>

          <label>
            User
            <select value={loginUserId} onChange={(event) => setLoginUserId(event.target.value)}>
              {(bootstrap?.users ?? []).map((user) => (
                <option key={user.id} value={user.id}>
                  {user.name}
                </option>
              ))}
            </select>
          </label>

          {loginUser && (
            <div className="login-details">
              <div>
                <KeyRound size={16} />
                <span>{loginUser.roles.join(", ")}</span>
              </div>
              <div>
                <ShieldCheck size={16} />
                <span>{loginUser.read_teams.join(", ")}</span>
              </div>
            </div>
          )}

          <button className="primary login-button" type="button" onClick={login} disabled={!bootstrap?.users.length} title="Continue">
            <LogIn size={17} />
            Continue
          </button>
          {error && <span className="error">{error}</span>}
        </section>
      </main>
    );
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="mark">DM</div>
          <div>
            <h1>DataMeta</h1>
            <p>Shoppy</p>
          </div>
        </div>

        <div className="active-user">
          <span className="field-label">Signed in as</span>
          <strong>{activeUser?.name}</strong>
          <small>{activeUser?.id}</small>
        </div>

        <div className="identity">
          <KeyRound size={16} />
          <span>{activeUser?.roles.join(", ")}</span>
        </div>
        <div className="identity">
          <ShieldCheck size={16} />
          <span>{activeUser?.read_teams.join(", ")}</span>
        </div>

        <nav className="tabs" aria-label="DataMeta views">
          <TabButton active={tab === "calculate"} icon={<SlidersHorizontal size={17} />} label="Calculate" onClick={() => setTab("calculate")} />
          <TabButton active={tab === "query"} icon={<MessageSquareText size={17} />} label="Query" onClick={() => setTab("query")} />
          <TabButton active={tab === "author"} icon={<SquarePen size={17} />} label="Author" onClick={() => setTab("author")} />
          <TabButton active={tab === "pipeline"} icon={<BarChart3 size={17} />} label="Pipeline" onClick={() => setTab("pipeline")} />
          <TabButton active={tab === "outliers"} icon={<AlertTriangle size={17} />} label="Outliers" onClick={() => setTab("outliers")} />
          <TabButton active={tab === "history"} icon={<History size={17} />} label="History" onClick={() => setTab("history")} />
        </nav>

        <button className="icon-text ghost" type="button" onClick={() => refresh()} title="Refresh">
          <RefreshCw size={16} />
          Refresh
        </button>
        <button className="icon-text ghost" type="button" onClick={switchUser} title="Switch user">
          <LogIn size={16} />
          Switch user
        </button>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">{modelStatus(bootstrap?.models)}</p>
            <h2>{titleForTab(tab)}</h2>
          </div>
          <div className="status-line">
            {loading && <Loader2 className="spin" size={18} />}
            {notice && <span className="notice">{notice}</span>}
            {error && <span className="error">{error}</span>}
          </div>
        </header>

        {tab === "calculate" && (
          <section className="view">
            <div className="command-row">
              <input value={calcQuestion} onChange={(event) => setCalcQuestion(event.target.value)} />
              <button className="primary" type="button" onClick={prepareCalculation} title="Find definitions">
                <Search size={17} />
                Find
              </button>
            </div>

            {prepare && (
              <>
                <div className="message-band">
                  <Sparkles size={18} />
                  <span>{prepare.message}</span>
                </div>
                <div className="definition-grid">
                  {prepare.definitions.map((definition) => (
                    <button
                      className={`definition-tile ${selectedDefinition === definition.id ? "selected" : ""}`}
                      key={definition.id}
                      type="button"
                      onClick={() => {
                        setSelectedDefinition(definition.id);
                        setSelectedTable(definition.accessible_tables[0]?.table ?? "");
                      }}
                    >
                      <div className="tile-head">
                        <span className={`team-dot ${definition.team}`} />
                        <strong>{definition.title}</strong>
                      </div>
                      <p>{definition.summary}</p>
                      <ul>
                        {definition.entails.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                      <small>
                        {definition.citation.path} · {definition.citation.commit}
                      </small>
                    </button>
                  ))}
                </div>

                <div className="runner-row">
                  <label>
                    Definition
                    <select value={selectedDefinition} onChange={(event) => setSelectedDefinition(event.target.value)}>
                      {prepare.definitions.map((definition) => (
                        <option key={definition.id} value={definition.id}>
                          {definition.team} · {definition.scope}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Accessible table
                    <select value={selectedTable} onChange={(event) => setSelectedTable(event.target.value)}>
                      {tableOptions.map((table) => (
                        <option key={table.table} value={table.table}>
                          {table.table}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button className="primary" type="button" onClick={runCalculation} disabled={!selectedDefinition || !selectedTable} title="Run calculation">
                    <Play size={17} />
                    Run
                  </button>
                </div>
              </>
            )}

            {calculation && (
              <div className="result-band">
                <div>
                  <p className="eyebrow">{calculation.definition?.team} · {calculation.table}</p>
                  <h3>{money(calculation.result?.value)}</h3>
                </div>
                <code>{calculation.sql}</code>
              </div>
            )}
          </section>
        )}

        {tab === "query" && (
          <section className="view">
            <div className="command-row">
              <input value={query} onChange={(event) => setQuery(event.target.value)} />
              <button className="primary" type="button" onClick={askQuestion} title="Ask">
                <MessageSquareText size={17} />
                Ask
              </button>
            </div>
            <label className="toggle">
              <input type="checkbox" checked={includeTrace} onChange={(event) => setIncludeTrace(event.target.checked)} />
              Trace
            </label>
            {answer && (
              <div className="answer">
                <p>{String(answer.answer)}</p>
                <CitationList citations={(answer.citations as Citation[]) ?? []} />
                {Boolean(answer.trace) && (
                  <details>
                    <summary>Trace</summary>
                    <pre>{JSON.stringify(answer.trace, null, 2)}</pre>
                  </details>
                )}
              </div>
            )}
          </section>
        )}

        {tab === "author" && (
          <section className="view split">
            <div className="edit-pane">
              <label>
                Team
                <select value={targetTeam} onChange={(event) => setTargetTeam(event.target.value)}>
                  <option value="finance">finance</option>
                  <option value="renewals">renewals</option>
                  <option value="ops">ops</option>
                  <option value="analytics">analytics</option>
                  <option value="data-ownership">data-ownership</option>
                </select>
              </label>
              <textarea value={authorText} onChange={(event) => setAuthorText(event.target.value)} />
              <button className="primary" type="button" onClick={draftProposal} title="Draft proposal">
                <SquarePen size={17} />
                Draft
              </button>
            </div>
            <div className="output-pane">
              {proposal ? (
                <>
                  <div className="checks">
                    {proposal.validation.checks.map((check) => (
                      <div className={`check ${check.ok ? "ok" : "warn"}`} key={check.name}>
                        {check.ok ? <Check size={16} /> : <AlertTriangle size={16} />}
                        <span>{check.name}</span>
                        <small>{check.detail}</small>
                      </div>
                    ))}
                  </div>
                  <pre className="markdown-preview">{proposal.markdown}</pre>
                  <div className="button-row">
                    <button className="secondary" type="button" onClick={() => commitProposal(false)} title="Check commit">
                      <GitCommitHorizontal size={17} />
                      Commit
                    </button>
                    {proposal.validation.needs_confirmation && (
                      <button className="danger" type="button" onClick={() => commitProposal(true)} title="Confirm overwrite">
                        <AlertTriangle size={17} />
                        Confirm
                      </button>
                    )}
                  </div>
                  {commitResult && <pre className="json-preview">{JSON.stringify(commitResult, null, 2)}</pre>}
                </>
              ) : (
                <Empty icon={<BookOpenText size={22} />} label="No proposal" />
              )}
            </div>
          </section>
        )}

        {tab === "pipeline" && (
          <section className="view">
            <div className="toolbar">
              <button className="primary" type="button" onClick={runPipeline} title="Run pipeline">
                <Play size={17} />
                Run GMV Ranker
              </button>
              <span className="pill">orders · net_gmv · Shoppy colors</span>
            </div>
            {pipeline && (
              <>
                <div className="bars">
                  {pipeline.chart.map((item) => (
                    <div className="bar-row" key={item.label}>
                      <span>{item.label}</span>
                      <div className="bar-track">
                        <div className="bar-fill" style={{ width: `${Math.max(4, (item.value / maxChart) * 100)}%`, background: item.color }} />
                      </div>
                      <strong>{money(item.value)}</strong>
                    </div>
                  ))}
                </div>
                <DataTable rows={pipeline.table} />
                <code className="sql-line">{pipeline.sql}</code>
              </>
            )}
          </section>
        )}

        {tab === "outliers" && (
          <section className="view split">
            <div className="edit-pane">
              <label>
                Subject
                <input value={flagSubject} onChange={(event) => setFlagSubject(event.target.value)} />
              </label>
              <label>
                Description
                <textarea value={flagDescription} onChange={(event) => setFlagDescription(event.target.value)} />
              </label>
              <button className="primary" type="button" onClick={flagOutlier} title="Flag outlier">
                <AlertTriangle size={17} />
                Flag
              </button>
              <label>
                Resolution
                <textarea value={resolutionText} onChange={(event) => setResolutionText(event.target.value)} />
              </label>
            </div>
            <div className="output-pane">
              {(bootstrap?.history.flags ?? []).map((flag) => (
                <div className="flag-row" key={flag.id}>
                  <div>
                    <strong>{flag.subject}</strong>
                    <p>{flag.description}</p>
                    <small>{flag.status} · {flag.owner_team} · {dateLabel(flag.created_at)}</small>
                  </div>
                  {flag.status !== "resolved" && (
                    <button className="secondary compact" type="button" onClick={() => resolveFlag(flag.id)} title="Resolve">
                      <Check size={16} />
                      Resolve
                    </button>
                  )}
                </div>
              ))}
              {flagResult && <pre className="json-preview">{JSON.stringify(flagResult, null, 2)}</pre>}
            </div>
          </section>
        )}

        {tab === "history" && (
          <section className="view history-grid">
            <div>
              <h3><FileClock size={18} /> Documents</h3>
              <div className="list">
                {bootstrap?.history.documents.map((doc) => (
                  <div className="doc-row" key={doc.path}>
                    <span className={`team-dot ${doc.team}`} />
                    <div>
                      <strong>{doc.title}</strong>
                      <p>{doc.path}</p>
                    </div>
                    <ChevronRight size={16} />
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h3><GitCommitHorizontal size={18} /> Commits</h3>
              <div className="list">
                {bootstrap?.history.commits.map((commit) => (
                  <div className="commit-row" key={commit.hash}>
                    <code>{commit.short_hash}</code>
                    <div>
                      <strong>{commit.subject}</strong>
                      <p>{commit.author} · {dateLabel(commit.authored_at)}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h3><Database size={18} /> Tables</h3>
              <div className="list">
                {bootstrap?.schema.map((table) => (
                  <div className="schema-row" key={table.table}>
                    <strong>{table.table}</strong>
                    <p>{table.columns.join(", ")}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
      </section>
    </main>
  );
}

function TabButton({ active, icon, label, onClick }: { active: boolean; icon: React.ReactNode; label: string; onClick: () => void }) {
  return (
    <button className={`tab ${active ? "active" : ""}`} type="button" onClick={onClick} title={label}>
      {icon}
      {label}
    </button>
  );
}

function titleForTab(tab: Tab) {
  const titles: Record<Tab, string> = {
    calculate: "ARR Calculation",
    query: "Grounded Query",
    author: "Knowledge Authoring",
    pipeline: "Runbook Runner",
    outliers: "Outlier Review",
    history: "Git History"
  };
  return titles[tab];
}

function CitationList({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <div className="citations">
      {citations.map((citation) => (
        <span key={`${citation.path}-${citation.commit}`}>
          {citation.path} · {citation.commit}
        </span>
      ))}
    </div>
  );
}

function DataTable({ rows }: { rows: Array<{ category: string; net_gmv: number; order_count: number }> }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Category</th>
          <th>Net GMV</th>
          <th>Orders</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.category}>
            <td>{row.category}</td>
            <td>{money(row.net_gmv)}</td>
            <td>{row.order_count}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Empty({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="empty">
      {icon}
      <span>{label}</span>
    </div>
  );
}
