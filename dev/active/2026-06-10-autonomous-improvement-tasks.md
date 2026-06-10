# DataMeta Autonomous Improvement — Task List

**Last Updated:** 2026-06-10 · All planned work complete; only follow-ups remain.

## Completed this session

- ✅ Diagnose 4 failing backend tests → root cause: merged branch left tests asserting pre-merge corpus (6 repos / 99.72%) while seed had 9 repos + duplicated storyline (see context doc)
- ✅ Fix seed corpus contradictions: deleted 3 gen1 docs (incident record, duplicate measurement, corrective actions), standardized on 99.62% / Sev-2 / 09:12–12:20 / failover-defect facts; harmonized warehouse SQLite rows; risk-register "Sev-1" wording neutralized; duplicate-seed-path guard added
- ✅ Fix retrieval routing regression: new `_rank_repositories` (repo metadata score blended with best file evidence ≥3.0); demo query routes to legal + platform-ops + customer-success again
- ✅ Fix unanswerable-query leak introduced by the blend (minimum_file_evidence=3.0)
- ✅ Update 4 stale tests to invariant-style assertions; counts now 9/21/63/71
- ✅ Add ARR conflict regression tests (headline demo had zero coverage): pause + 684k/700k/790k values + RBAC no-pause variant
- ✅ Neo4j sync prune (Repository/Folder/Document/Chunk DETACH DELETE for ids absent from index) + test
- ✅ Neo4j evidence: no-guessing entity filter (+ distractor row in mocked test) and local-score blend re-rank → live answer now includes 99.62% & 99.90%
- ✅ FastAPI `on_event` → lifespan migration
- ✅ Docs: README intro, demo-cheatsheet Act 3, demo-questions multi-repo section aligned with reality
- ✅ Surgical runtime cleanup (commit `d8c3bae` in runtime repo) + warehouse re-seed + live Aura re-sync with prune
- ✅ Full suite green: 24/24; 6 single-repo demo questions spot-checked OK
- ✅ Session report delivered to user; memory saved (`datameta-improvement-2026-06.md` in Claude memory dir)

## Pending / next session

- [ ] **Commit the 6 modified files** if user approves (they were asked; no answer yet). Do NOT commit `decks/`.
- [ ] (Optional) Improve doc-level ranking so "What tier is Vendor X?" top citation is `vendor-risk-management/risk-registers/vendor-tiering.md` rather than security's severity doc
- [ ] (Optional) Review README below the Run section + frontend for staleness
- [ ] (Optional) Migrate Aura Cypher off deprecated `db.index.vector.queryNodes` → `SEARCH` syntax
