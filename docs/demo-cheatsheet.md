# DataMeta — Demo Cheatsheet

A runnable script for demoing DataMeta. Internal use. Pair with
[`demo-questions.md`](./demo-questions.md) for the full question list.

---

## 0. Pre-demo setup (do this before the room is watching)

**Start the backend** (port 8000):
```
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Start the frontend** (port 3000 — backend CORS expects 3000):
```
cd frontend
npm run dev
```

Open **http://localhost:3000**. Confirm the login screen shows and the model
line reads "OpenAI configured".

**Smoke test (30 sec):** sign in as **Ada Admin** → Files shows repositories →
Query "What are our incident severity levels?" returns an answer. If yes, you're ready.

> If the page looks empty, your browser is probably on an old port (3001/3002).
> Use **3000**.

---

## 1. The one-liner

> "DataMeta is a knowledge layer over an enterprise's markdown repos. It answers
> questions with citations, respects who's allowed to see what, and — crucially —
> when teams disagree on a definition, it **stops and asks you which one** instead
> of guessing."

---

## 2. Demo flow (≈6 minutes, 4 acts)

### Act 1 — Browse like GitHub (Files tab) · *user: Ada Admin*
- Click **Files**. You see **repositories**, not a wall of files.
- Click a repo → **folders**; click a folder → **files**; click a file → it opens.
- **Say:** "Knowledge is organized repo → folder → file, like GitHub. Each level
  has its own metadata."

### Act 2 — Single-repo answer (Query tab) · *user: Ada Admin*
- **Query** → type: **"What service credit do customers get for an SLA miss?"**
- One clean answer with a citation.
- **Say:** "That answer came from one repo — Legal. Flip on **Trace** to see the
  retrieval funnel."

### Act 3 — Multi-repo synthesis (Query tab) · *user: Ada Admin*
- **Query** → type:
  **"Customer A says we missed their SLA during the Vendor X outage. What should we do?"**
- **Say:** "This needs several teams at once — Legal for the 99.90% SLA terms,
  Platform Ops for the measured 99.62% availability and the incident record,
  Customer Success for the complaint and escalation playbook, Vendor Risk for
  the Vendor X RCA, Data Governance for the evidence pack. It stitches them
  together and cites every source. If the docs don't support an answer, it says
  so rather than inventing one."

### Act 4 — ⭐ The conflict (the headline) · *user: Olivia Operations Associate*
- **Switch user** → sign in as **Olivia Operations Associate** (`ops.associate`).
- **Query** → type: **"Help me calculate ARR for ASEAN"**
- The app **pauses** and shows **three ARR definition cards** with
  *"Multiple ARR definitions are visible — choose one."*
- **Say:** "Finance, Renewals, and Sales each define ARR differently. The system
  won't guess — it asks, like Claude Code would."
- Click **Finance** → **$684,000** + the exact SQL it ran.
- Click **Renewals** → **$700,000**. Click **Sales** → **$790,000**.
- **Say:** "Same question, three legitimate answers. The point is governance —
  you always know *whose* definition produced the number, with the SQL to prove it."

| Definition | Logic | ARR (ASEAN) |
|-----------|-------|-------------|
| Finance (board) | active MRR × 12 | **$684,000** |
| Renewals (forecast) | committed + likely renewals | **$700,000** |
| Sales (presentation) | all booked ACV incl. pipeline | **$790,000** |

---

## 3. Optional add-ons (if there's time / interest)

- **RBAC:** Sign in as **Leah Legal** and ask the ARR question → no conflict,
  because she can't see the Finance/Renewals/Sales repos. "Access controls the
  answer, not just the file list."
- **Authoring:** **Author** tab → drafts a knowledge proposal and checks it for
  conflicts before committing.
- **MCP:** The same brain is exposed as an MCP server at
  `http://127.0.0.1:8000/mcp` (16 tools). An agent calling `datameta_ask` gets
  the *same* ARR pause — "this isn't just a UI, it's an agent-ready tool."

---

## 4. Quick user reference

| Sign-in | Sees | Use for |
|---------|------|---------|
| **Ada Admin** (`ada.admin`) | everything | Acts 1–3, full walkthrough |
| **Olivia Operations Associate** (`ops.associate`) | finance, renewals, sales | Act 4 (ARR conflict) |
| **Jamie Incident Analyst** (`junior.analyst`) | all 6 incident repos | alt for Acts 1–3 |
| **Leah Legal** (`leah.legal`) | legal + a few | RBAC contrast |

---

## 5. If something breaks (fallbacks)

- **Login button greyed out / no data:** backend not reachable, or browser on the
  wrong port. Confirm backend is up on 8000 and you're on **localhost:3000**.
- **ARR question doesn't pause:** make sure you're signed in as **Olivia** (or
  Ada) — other users can't see all three ARR repos, so there's no conflict.
- **Need a clean slate:** stop the backend, delete the `runtime/` folder, restart
  — it re-seeds all data fresh.
- **Worst case:** screenshots of each act in your back pocket.

---

## 6. The three sentences to land

1. "It answers from our own repos, with citations, and won't make things up."
2. "It respects who's allowed to see what."
3. "When teams disagree, it asks which definition you mean instead of guessing —
   and shows the SQL behind the number."
