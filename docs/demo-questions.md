# DataMeta — Demo Question Sheet (internal)

Cheat-sheet for running the demo. Not shipped to end users.

All questions are typed into the **Query** box. The Query box is now a single
"ask anything" entry point that routes automatically:

- a normal knowledge question → RAG answer with citations;
- a metric question with conflicting definitions → **pauses and asks which
  definition** before computing.

---

## 1. Single-repository questions

Answer comes primarily from **one** repo. Sign in as **Ada Admin** (sees
everything) or **Jamie Incident Analyst**.

| Question | Primary repo |
|----------|--------------|
| What are our incident severity levels? | security-incident-response |
| What service credit do customers get for an SLA miss? | legal-contracts |
| How long do we keep incident evidence? | data-governance |
| What tier is Vendor X? | vendor-risk-management |
| How do we measure availability / what is our SLO? | platform-operations |
| Who owns the Customer A relationship? | customer-success-ops |

> Note: the retrieval funnel always *shortlists* up to 4 repos; the **citations**
> show the answer really came from one. Use the Trace toggle to show the funnel.

## 2. Multi-repository question

Requires synthesis across repos. Sign in as **Ada Admin** or **Jamie**.

- **"Customer A says we missed their SLA during the Vendor X outage. What should we do?"**
  Pulls legal (SLA terms) + platform-operations (incident + 99.62% measured
  availability + postmortem) + vendor-risk (Vendor X RCA) + security
  (customer notice) + data-governance (evidence pack).

## 3. ⭐ Cross-repo CONFLICT — the headline demo

Three teams define **ARR** differently, in three different repos:

| Repo / team | Definition | Formula (on `arr_subscriptions`, ASEAN) | ARR |
|-------------|-----------|------------------------------------------|-----|
| **Finance** (board) | active MRR × 12 | `SUM(mrr)*12 WHERE status='active'` | **$684,000** |
| **Renewals** (forecast) | committed + likely renewals | `SUM(acv) WHERE renewal_status IN ('committed','likely')` | **$700,000** |
| **Sales** (exec presentation) | all booked ACV incl. pipeline | `SUM(acv) WHERE status != 'churned'` | **$790,000** |

**Steps:**
1. Sign in as **Olivia Operations Associate** (`ops.associate`) — she can read
   Finance, Renewals, and Sales.
2. In **Query**, type: **"Help me calculate ARR for ASEAN"**.
3. The app does **not** guess. It pauses and shows three definition cards with
   the message *"Multiple ARR definitions are visible. Choose the one that
   matches your business scope."*
4. Click a card (e.g. **Finance**) → it runs that team's formula and shows the
   number **+ the exact SQL** used.
5. Click a different card → a different number. That's the point: the answer
   depends on whose definition you mean, and the system makes you choose.

**RBAC variant:** sign in as a user who can read only one of the three (or who
lacks the `arr_subscriptions` table) and ask the same question — no conflict
appears, demonstrating access-scoped behavior.

---

### Users
| Sign-in | Sees |
|---------|------|
| `ada.admin` (Ada Admin) | everything (all 9 repos + all tables) — best for a full walkthrough |
| `junior.analyst` (Jamie) | all 6 incident repos |
| `ops.associate` (Olivia) | finance, renewals, sales (ARR conflict) |
| `leah.legal`, `cam.cs`, … | scoped subsets — good for RBAC contrasts |
