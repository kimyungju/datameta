"use client";

import {
  AlertTriangle,
  BookOpenText,
  Check,
  ChevronRight,
  Database,
  FileClock,
  FileText,
  Folder,
  GitCommitHorizontal,
  History,
  KeyRound,
  LogIn,
  Loader2,
  MessageSquareText,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
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
  repository?: string;
  folder?: string;
  file_path?: string;
  heading?: string;
  snippet?: string;
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
  inventory?: Record<string, unknown>;
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

type ConflictFile = {
  id: string;
  path: string;
  team: string;
  entity: string;
  scope: string;
  title: string;
  summary: string;
  markdown: string;
  citation: Citation;
};

type ValidationResult = {
  ok: boolean;
  blocked: boolean;
  needs_confirmation: boolean;
  checks: Array<{
    name: string;
    ok: boolean;
    detail: string;
    conflicts?: ConflictFile[];
    matching_tables?: unknown[];
  }>;
  samples: Record<string, Record<string, unknown[]>>;
};

type ApiError = { detail?: string };

type Tab = "query" | "files" | "author" | "history";

type ShortlistItem = {
  id: string;
  level: string;
  repository: string;
  folder: string;
  path: string;
  title: string;
  summary: string;
  hybrid?: {
    score?: number;
    keyword_score?: number;
    vector_score?: number;
    phrase_boost?: number;
    keyword_hits?: string[];
  };
};

type FolderAgentResult = {
  repository: string;
  folder: string;
  folder_path: string;
  agent: string;
  agent_error?: string;
  selected_files: ShortlistItem[];
  full_content_files_read: string[];
  findings: Array<{
    file_path: string;
    title: string;
    heading: string;
    summary: string;
    snippet: string;
    citation: Citation;
  }>;
};

type MultirepoAnswer = {
  answerable: boolean;
  answer: string;
  citations: Citation[];
  shortlisted_repositories: ShortlistItem[];
  shortlisted_folders: ShortlistItem[];
  shortlisted_files: ShortlistItem[];
  folder_subagent_findings: FolderAgentResult[];
  trace?: Record<string, unknown>;
};

type MarkdownFileSummary = {
  score: number;
  keyword_hits: string[];
  id: string;
  path: string;
  repository: string;
  folder: string;
  metadata_level: string;
  team: string;
  type: string;
  title: string;
  summary: string;
  customers: string[];
  vendors: string[];
  heading: string;
  snippet: string;
  citation: Citation;
};

type MarkdownSearchResponse = {
  query: string;
  files: MarkdownFileSummary[];
  total: number;
};

type MarkdownFileDetail = {
  id: string;
  path: string;
  repository: string;
  folder: string;
  metadata_level: string;
  team: string;
  type: string;
  title: string;
  summary: string;
  metadata: Record<string, string>;
  body: string;
  markdown: string;
  citation: Citation;
};

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
  const [tab, setTab] = useState<Tab>("query");
  const [userId, setUserId] = useState("");
  const [loginUserId, setLoginUserId] = useState("junior.analyst");
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  const [prepare, setPrepare] = useState<PrepareResult | null>(null);
  const [selectedDefinition, setSelectedDefinition] = useState("");
  const [selectedTable, setSelectedTable] = useState("");
  const [calculation, setCalculation] = useState<CalculationResult | null>(null);

  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState<MultirepoAnswer | null>(null);
  const [includeTrace, setIncludeTrace] = useState(true);
  const [fileSearch, setFileSearch] = useState("");
  const [submittedSearch, setSubmittedSearch] = useState("");
  const [browseRepo, setBrowseRepo] = useState<string | null>(null);
  const [browseFolder, setBrowseFolder] = useState<string | null>(null);
  const [fileResults, setFileResults] = useState<MarkdownFileSummary[]>([]);
  const [selectedFile, setSelectedFile] = useState<MarkdownFileDetail | null>(null);

  const [authorText, setAuthorText] = useState("");
  const [targetTeam, setTargetTeam] = useState("");
  const [proposal, setProposal] = useState<ProposalResult | null>(null);
  const [commitResult, setCommitResult] = useState<Record<string, unknown> | null>(null);

  const activeUser = useMemo(() => bootstrap?.users.find((user) => user.id === userId), [bootstrap, userId]);
  const loginUser = useMemo(() => bootstrap?.users.find((user) => user.id === loginUserId), [bootstrap, loginUserId]);
  const writableTeams = useMemo(() => activeUser?.write_teams ?? [], [activeUser]);

  useEffect(() => {
    if (!writableTeams.includes(targetTeam)) setTargetTeam(writableTeams[0] ?? "");
  }, [writableTeams, targetTeam]);

  const repositories = useMemo(() => {
    const map = new Map<string, { name: string; title: string; summary: string; folders: Set<string>; fileCount: number }>();
    for (const file of fileResults) {
      let entry = map.get(file.repository);
      if (!entry) {
        entry = { name: file.repository, title: file.repository, summary: "", folders: new Set(), fileCount: 0 };
        map.set(file.repository, entry);
      }
      entry.fileCount += 1;
      if (file.folder) entry.folders.add(file.folder);
      if (file.metadata_level === "repository") {
        entry.title = file.title || file.repository;
        entry.summary = file.summary || "";
      }
    }
    return Array.from(map.values()).sort((a, b) => a.name.localeCompare(b.name));
  }, [fileResults]);

  const repoFiles = useMemo(
    () => (browseRepo ? fileResults.filter((file) => file.repository === browseRepo) : []),
    [fileResults, browseRepo]
  );

  const repoRootFiles = useMemo(() => repoFiles.filter((file) => !file.folder), [repoFiles]);

  const folders = useMemo(() => {
    const map = new Map<string, { name: string; title: string; summary: string; fileCount: number }>();
    for (const file of repoFiles) {
      if (!file.folder) continue;
      let entry = map.get(file.folder);
      if (!entry) {
        entry = { name: file.folder, title: file.folder, summary: "", fileCount: 0 };
        map.set(file.folder, entry);
      }
      entry.fileCount += 1;
      if (file.metadata_level === "folder") {
        entry.title = file.title || file.folder;
        entry.summary = file.summary || "";
      }
    }
    return Array.from(map.values()).sort((a, b) => a.name.localeCompare(b.name));
  }, [repoFiles]);

  const folderFiles = useMemo(
    () => (browseFolder ? repoFiles.filter((file) => file.folder === browseFolder) : []),
    [repoFiles, browseFolder]
  );

  const currentRepo = useMemo(() => repositories.find((repo) => repo.name === browseRepo), [repositories, browseRepo]);
  const currentFolder = useMemo(() => folders.find((folder) => folder.name === browseFolder), [folders, browseFolder]);

  function renderFileRow(file: MarkdownFileSummary) {
    return (
      <button
        className={`file-row ${selectedFile?.path === file.path ? "selected" : ""}`}
        key={file.path}
        type="button"
        onClick={() => openMarkdownFile(file.path)}
      >
        <FileText size={16} className="browse-icon" />
        <div>
          <strong>{file.title}</strong>
          <p>{file.path}</p>
          <small>
            {file.repository} · {file.folder || "repository"} · {file.metadata_level} · {file.citation.commit}
          </small>
        </div>
        <ChevronRight size={16} />
      </button>
    );
  }

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
    setFileResults([]);
    setSubmittedSearch("");
    setBrowseRepo(null);
    setBrowseFolder(null);
    setSelectedFile(null);
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
    setFileResults([]);
    setSubmittedSearch("");
    setBrowseRepo(null);
    setBrowseFolder(null);
    setSelectedFile(null);
    setProposal(null);
    setCommitResult(null);
    setNotice("");
  }

  async function askQuestion() {
    if (!query.trim() || loading) return;
    const result = await runAction(() =>
      api<Record<string, unknown>>("/api/ask", {
        method: "POST",
        body: JSON.stringify({ question: query, include_trace: includeTrace })
      })
    );
    if (!result) return;
    const intent = result.intent as string | undefined;
    if (intent === "choose_definition") {
      const prep = result as unknown as PrepareResult;
      setPrepare(prep);
      const first = prep.definitions[0];
      setSelectedDefinition(first?.id ?? "");
      setSelectedTable(first?.accessible_tables[0]?.table ?? "");
      setCalculation(null);
      setAnswer(null);
    } else if (intent === "calculation") {
      setPrepare((result.prepare as PrepareResult) ?? null);
      setCalculation((result.calculation as CalculationResult) ?? null);
      setAnswer(null);
    } else {
      setAnswer(result as unknown as MultirepoAnswer);
      setPrepare(null);
      setCalculation(null);
    }
  }

  async function chooseDefinition(definitionId: string, table: string) {
    setSelectedDefinition(definitionId);
    setSelectedTable(table);
    const result = await runAction(
      () =>
        api<CalculationResult>("/api/calculation/run", {
          method: "POST",
          body: JSON.stringify({ definition_id: definitionId, table })
        }),
      "Calculation complete"
    );
    if (result) setCalculation(result);
  }

  async function searchMarkdown(nextQuery = fileSearch) {
    const result = await runAction(() =>
      api<MarkdownSearchResponse>("/api/markdown/search", {
        method: "POST",
        body: JSON.stringify({ query: nextQuery, limit: 100 })
      })
    );
    if (result) {
      setFileResults(result.files);
      setSubmittedSearch(nextQuery.trim());
      setBrowseRepo(null);
      setBrowseFolder(null);
      if (!result.files.some((file) => file.path === selectedFile?.path)) setSelectedFile(null);
    }
  }

  async function openMarkdownFile(path: string) {
    const result = await runAction(() =>
      api<MarkdownFileDetail>("/api/markdown/read", {
        method: "POST",
        body: JSON.stringify({ path })
      })
    );
    if (result) setSelectedFile(result);
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


  if (!userId) {
    return (
      <main className="login-shell">
        <section className="login-panel">
          <div className="brand login-brand">
            <div className="mark">DM</div>
            <div>
              <h1>DataMeta</h1>
              <p>Generic Enterprise</p>
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
              <p>Generic Enterprise</p>
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
          <TabButton active={tab === "query"} icon={<MessageSquareText size={17} />} label="Query" onClick={() => setTab("query")} />
          <TabButton
            active={tab === "files"}
            icon={<FileText size={17} />}
            label="Files"
            onClick={() => {
              setTab("files");
              if (!fileResults.length) searchMarkdown().catch(() => undefined);
            }}
          />
          <TabButton active={tab === "author"} icon={<SquarePen size={17} />} label="Author" onClick={() => setTab("author")} />
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

        {tab === "query" && (
          <section className="view">
            <div className="command-row">
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") askQuestion().catch(() => undefined);
                }}
                placeholder="Ask a question..."
              />
              <button className="primary" type="button" onClick={askQuestion} disabled={loading || !query.trim()} title="Ask">
                <MessageSquareText size={17} />
                Ask
              </button>
            </div>
            <label className="toggle">
              <input type="checkbox" checked={includeTrace} onChange={(event) => setIncludeTrace(event.target.checked)} />
              Trace
            </label>

            {prepare && prepare.requires_choice && (
              <>
                <div className="message-band warn">
                  <SlidersHorizontal size={18} />
                  <span>{prepare.message}</span>
                </div>
                <div className="definition-grid">
                  {prepare.definitions.map((definition) => {
                    const table = definition.accessible_tables[0]?.table ?? "";
                    return (
                      <button
                        className={`definition-tile ${selectedDefinition === definition.id ? "selected" : ""}`}
                        key={definition.id}
                        type="button"
                        onClick={() => chooseDefinition(definition.id, table)}
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
                        <small>Use {definition.team} definition · table {table || "—"}</small>
                      </button>
                    );
                  })}
                </div>
              </>
            )}

            {calculation && calculation.result && (
              <div className="result-band">
                <div>
                  <p className="eyebrow">{calculation.definition?.team} · {calculation.table}</p>
                  <h3>{money(calculation.result.value)}</h3>
                </div>
                <code>{calculation.sql}</code>
              </div>
            )}

            {answer && (
              <div className="answer">
                <div className={`message-band ${answer.answerable ? "" : "warn"}`}>
                  {answer.answerable ? <Check size={18} /> : <AlertTriangle size={18} />}
                  <span>{answer.answerable ? "Answerable from selected markdown" : "Not answerable from available knowledge"}</span>
                </div>
                <p className="answer-text">{answer.answer}</p>
                <TraceSections result={answer} />
                <CitationList citations={answer.citations ?? []} />
                {Boolean(answer.trace) && (
                  <details>
                    <summary>Raw trace</summary>
                    <pre>{JSON.stringify(answer.trace, null, 2)}</pre>
                  </details>
                )}
              </div>
            )}
          </section>
        )}

        {tab === "files" && (
          <section className="view files-view">
            <div className="command-row">
              <input
                value={fileSearch}
                onChange={(event) => setFileSearch(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") searchMarkdown().catch(() => undefined);
                }}
                placeholder="Search markdown files"
              />
              <button className="primary" type="button" onClick={() => searchMarkdown()} title="Search files">
                <Search size={17} />
                Search
              </button>
            </div>

            <div className="files-browser">
              <div className="list file-list">
                {submittedSearch ? (
                  fileResults.length ? (
                    fileResults.map((file) => renderFileRow(file))
                  ) : (
                    <Empty icon={<FileText size={22} />} label="No files match your search" />
                  )
                ) : (
                  <>
                    <div className="browse-bar">
                      <nav className="breadcrumb" aria-label="File location">
                        <button
                          type="button"
                          onClick={() => {
                            setBrowseRepo(null);
                            setBrowseFolder(null);
                          }}
                        >
                          Repositories
                        </button>
                        {browseRepo && (
                          <>
                            <ChevronRight size={14} />
                            <button type="button" onClick={() => setBrowseFolder(null)}>
                              {currentRepo?.title ?? browseRepo}
                            </button>
                          </>
                        )}
                        {browseFolder && (
                          <>
                            <ChevronRight size={14} />
                            <span>{currentFolder?.title ?? browseFolder}</span>
                          </>
                        )}
                      </nav>
                    </div>

                    {!browseRepo &&
                      (repositories.length ? (
                        repositories.map((repo) => (
                          <button
                            className="browse-row"
                            key={repo.name}
                            type="button"
                            onClick={() => {
                              setBrowseRepo(repo.name);
                              setBrowseFolder(null);
                            }}
                          >
                            <Database size={18} className="browse-icon" />
                            <div>
                              <strong>{repo.title}</strong>
                              {repo.summary && <p>{repo.summary}</p>}
                              <small>
                                {repo.name} · {repo.folders.size} folder{repo.folders.size === 1 ? "" : "s"}
                              </small>
                            </div>
                            <ChevronRight size={16} />
                          </button>
                        ))
                      ) : (
                        <Empty icon={<Database size={22} />} label="No repositories available" />
                      ))}

                    {browseRepo && !browseFolder && (
                      <>
                        {folders.map((folder) => (
                          <button
                            className="browse-row"
                            key={folder.name}
                            type="button"
                            onClick={() => setBrowseFolder(folder.name)}
                          >
                            <Folder size={18} className="browse-icon" />
                            <div>
                              <strong>{folder.title}</strong>
                              {folder.summary && <p>{folder.summary}</p>}
                              <small>
                                {folder.name} · {folder.fileCount} file{folder.fileCount === 1 ? "" : "s"}
                              </small>
                            </div>
                            <ChevronRight size={16} />
                          </button>
                        ))}
                        {repoRootFiles.map((file) => renderFileRow(file))}
                        {!folders.length && !repoRootFiles.length && (
                          <Empty icon={<FileText size={22} />} label="This repository is empty" />
                        )}
                      </>
                    )}

                    {browseRepo && browseFolder && (
                      folderFiles.length ? (
                        folderFiles.map((file) => renderFileRow(file))
                      ) : (
                        <Empty icon={<FileText size={22} />} label="This folder is empty" />
                      )
                    )}
                  </>
                )}
              </div>

              <div className="markdown-reader">
                {selectedFile ? (
                  <>
                    <div className="reader-head">
                      <div>
                        <p className="eyebrow">{selectedFile.repository} · {selectedFile.folder || "repository"}</p>
                        <h3>{selectedFile.title}</h3>
                        <small>{selectedFile.path} · {selectedFile.citation.commit}</small>
                      </div>
                    </div>
                    <pre className="markdown-preview">{selectedFile.markdown}</pre>
                  </>
                ) : (
                  <Empty icon={<BookOpenText size={22} />} label="Select a markdown file" />
                )}
              </div>
            </div>
          </section>
        )}

        {tab === "author" && (
          <section className="view split">
            <div className="edit-pane">
              {writableTeams.length ? (
                <>
                  <label>
                    Team
                    <select value={targetTeam} onChange={(event) => setTargetTeam(event.target.value)}>
                      {writableTeams.map((team) => (
                        <option key={team} value={team}>
                          {team}
                        </option>
                      ))}
                    </select>
                  </label>
                  <textarea value={authorText} onChange={(event) => setAuthorText(event.target.value)} placeholder="Draft an update..." />
                  <button
                    className="primary"
                    type="button"
                    onClick={draftProposal}
                    disabled={loading || !authorText.trim() || !targetTeam}
                    title="Draft proposal"
                  >
                    <SquarePen size={17} />
                    Draft
                  </button>
                </>
              ) : (
                <Empty
                  icon={<ShieldCheck size={22} />}
                  label={`${activeUser?.name ?? "This user"} has no write access to any team repository`}
                />
              )}
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
                        {check.conflicts && check.conflicts.length > 0 && (
                          <div className="conflict-files">
                            {check.conflicts.map((conflict) => (
                              <div className="conflict-file" key={conflict.id}>
                                <div className="conflict-file-head">
                                  <FileClock size={14} />
                                  <strong>
                                    {conflict.entity} · {conflict.scope}
                                  </strong>
                                </div>
                                <code className="conflict-file-path">{conflict.path}</code>
                                <p className="conflict-file-note">
                                  This committed file will be overwritten if you confirm.
                                </p>
                                <pre className="conflict-file-markdown">{conflict.markdown}</pre>
                              </div>
                            ))}
                          </div>
                        )}
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
    query: "Query",
    files: "Files",
    author: "Knowledge Authoring",
    history: "Git History"
  };
  return titles[tab];
}

function TraceSections({ result }: { result: MultirepoAnswer }) {
  return (
    <div className="trace-grid">
      <TraceList title="Repositories" items={result.shortlisted_repositories} />
      <TraceList title="Folders" items={result.shortlisted_folders} />
      <TraceList title="Files" items={result.shortlisted_files} />
      <div className="trace-panel">
        <h3>Folder Agents</h3>
        <div className="list">
          {result.folder_subagent_findings.map((folder) => (
            <div className="agent-row" key={folder.folder_path}>
              <div>
                <strong>{folder.folder_path}</strong>
                <p>{folder.agent} · read {folder.full_content_files_read.length} file(s)</p>
                {folder.agent_error && <small>{folder.agent_error}</small>}
              </div>
              {folder.findings.map((finding) => (
                <blockquote key={`${folder.folder_path}-${finding.file_path}-${finding.heading}`}>
                  <strong>{finding.title}</strong>
                  <p>{finding.snippet}</p>
                  <small>{finding.file_path} · {finding.citation.commit}</small>
                </blockquote>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function TraceList({ title, items }: { title: string; items: ShortlistItem[] }) {
  return (
    <div className="trace-panel">
      <h3>{title}</h3>
      <div className="list">
        {items.map((item) => (
          <div className="doc-row" key={`${title}-${item.path}`}>
            <div>
              <strong>{item.title}</strong>
              <p>{item.path}</p>
              <small>
                score {item.hybrid?.score?.toFixed?.(2) ?? "-"} · vector {item.hybrid?.vector_score?.toFixed?.(2) ?? "-"} · keyword{" "}
                {item.hybrid?.keyword_score?.toFixed?.(2) ?? "-"}
              </small>
            </div>
            <ChevronRight size={16} />
          </div>
        ))}
      </div>
    </div>
  );
}

function CitationList({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <div className="citations">
      {citations.map((citation) => (
        <span key={`${citation.path}-${citation.heading}-${citation.commit}`}>
          {citation.file_path ?? citation.path} · {citation.heading ?? citation.title} · {citation.commit}
        </span>
      ))}
    </div>
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
