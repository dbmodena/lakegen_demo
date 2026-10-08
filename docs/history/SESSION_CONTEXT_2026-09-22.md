# LakeGen session handoff — keyword retrieval & table selection work

Written 2026-09-22, for a fresh Claude Code session picking this up on the same
machine. Read this whole file before touching anything — several findings
below overturned earlier numbers in this same session, and the git history
does **not** capture most of the experimental work (see "What's committed").

## TL;DR state

- **Keyword search retrieval is fixed and shipped.** The banlist now shrinks
  to minimal failing word-subsets instead of banning whole queries, the OR
  fallback that silently bypassed it is gone, and a new
  `decompose_preview_search` pipeline (split question → per-table search
  phrases → preview real AND-word candidates → model picks before anything
  runs) is live in production, **on by default**.
- **Selection (picking final tables from the candidate pool) is the current
  bottleneck**, not retrieval. Quantified: ~26-30 point gap between "ceiling"
  (right tables were visible) and "exact" (agent actually picked them
  correctly) on every selection arm tested.
- **A 4-stage selection prototype has been proposed but not built yet** —
  that's the next work. See "Proposed selection prototype" below.
- `gpt-oss-20b` is now registered and live-verified reachable via OCI,
  specifically to test which pipeline stages tolerate a smaller/cheaper model
  and which need `gpt-oss-120b`'s reasoning strength.

## What's committed vs what's only on local disk

- **Committed** (`git log`, commit `4351673 "keyword search update"`):
  `src/lakegen/agent_tools/tools_p12.py`, `src/lakegen/experiment_config.py`,
  `src/lakegen/phases/phase12.py` — the production `decompose_preview_search`
  feature.
- **Uncommitted, tracked, modified right now**: `src/lakegen/core/resources.py`
  (the `gpt-oss-20b` registration, done this session, not yet committed).
  Also `scripts_pneuma/bootstrap_pneuma.py` shows modified — **not touched by
  this session's work**, that's separate/unrelated, don't assume it's mine.
- **Never tracked by git at all**: everything under `experiments/retrieval_lab/`
  — the entire retrieval/selection lab (`arms_retrieval.py`, `arms_select.py`,
  `arms_select2.py`, `common.py`, `orqa10.py`, `report_stats.py`,
  `rename_map.py`, `run_lab.py`, 103 result `.jsonl` files, the LLM response
  cache) is gitignored (`experiments/` in `.gitignore`). It's real, substantial,
  validated work — 100+ questions tested across a dozen arms — but it only
  exists on this machine's disk. If you need it elsewhere, `git add -f` the
  specific files you want.
- **`tests/` is also gitignored** (root `.gitignore`). Two new test files this
  session — `tests/test_decompose_preview_search.py`,
  `tests/test_distinguishing_prefilter.py` — exist on disk, pass, but are
  untracked. `git add -f` if you want them committed.

## Production code changes, in the order they happened

1. **Keyword banlist fix** (`src/lakegen/agent_tools/tools_p12.py`,
   `src/lakegen/keyword_terms.py`): a zero-result strict-AND query is now
   shrunk to its smallest failing word subsets (`minimal_failing_subsets`,
   real Solr counts via `retriever.lexical_match_count`) instead of banning
   the whole query; `search_keyword_concepts` splits concepts into
   Solr-equivalent words (`split_keywords`) before banning/searching, matching
   `WordDelimiterGraphFilter`; the OR-fallback that silently bypassed the
   banlist is removed; HYBRID mode now detects when the lexical branch matched
   nothing even though the semantic branch rescued the search
   (`retriever.last_lexical_hit_count`), and bans that too. Tests:
   `tests/test_ban_shrinking.py`, `tests/test_keyword_terms.py`.
   - **This got reverted once already** by a teammate's concurrent merge
     mid-session (commit `2e314d6`, legitimate v2→v1 promotion that used a
     stale pre-fix copy) and had to be reapplied from scratch. This is a live,
     shared codebase — expect concurrent edits, verify current file state
     with `grep`/`git log` before trusting anything described here as still
     present.

2. **`decompose_preview_search`** (`DiscoveryConfig.decompose_preview_search`,
   default `True`): ported from the retrieval lab's best-performing arm
   (`decompose_tuned_preview_noacronym`). When `search_keyword_concepts` runs
   and an `llm` is on the manager (threaded through from `phase12_agent`),
   it now: (a) one LLM call splits the *original* question into 1-4 per-table
   search phrases — one entry per distinguishing date/edition, never folding
   two into one, and never guessing/expanding an organisation's name; (b) for
   each phrase, extracts index-real content words (stoplist now also excludes
   quantifier/aggregation words — total/average/count/correlation/etc, added
   this session), pair-counts them against Solr, previews the top real
   matches, and one LLM call per phrase picks 1-3 word-sets to actually search;
   (c) each pick runs through the *same* retrieval/banlist/hybrid-lexical-zero
   path every search already used (refactored into `_run_one_retrieval` /
   `_search_limit_reached` / `_format_search_response`, reused by both the
   plain and decompose paths — verified byte-for-byte behavior-preserving via
   the full existing test suite before adding the new path).
   Falls back silently to the old single-shot search if `llm is None`, so it's
   always safe to leave on. Tests: `tests/test_decompose_preview_search.py`
   (5 tests, mocked). Also live-smoke-tested against real Solr + real LLM:
   correctly split "June 2022 and July 2023" into two separate table searches
   in 7.3s (the exact multi-date-collapse bug an earlier failed prototype hit
   — see "Dead ends" below).

3. **`gpt-oss-20b` registration** (`src/lakegen/core/resources.py`,
   **uncommitted**): added `OPENAI_GPT_OSS_20B_MODEL = "openai.gpt-oss-20b"`
   alongside the existing 120b constant, generalized the two places that
   special-cased the 120b string (max output tokens, context size kwarg —
   both share the family's 128k context), added it to `GENERIC_CHAT_MODELS`
   so streamed tool calls work. Live-verified: `get_llm("openai.gpt-oss-20b")`
   → real round trip through OCI, correct response. Motivation: test which
   selection-prototype stages need `gpt-oss-120b`'s reasoning and which hold
   up on a smaller/cheaper model — this is prep work, not yet used in an
   actual test run.

## Key empirical findings (numbers that took real compute to produce)

All from `experiments/retrieval_lab/results/*.jsonl`, scored with
`experiments/retrieval_lab/report_stats.py` (paired bootstrap 95% CI,
`boot_diff`). Re-derive with that machinery rather than trusting these numbers
forever — several already went stale once this session (see next section).

**Retrieval stage** (full@20 = did the retrieved top-20 pool contain every
gold table):
- `kw_baseline` (plain single-shot strict-AND search) vs `decompose_detail_preview`
  (split-by-distinguishing-detail + preview-and-pick, the mechanism now in
  production): **kw_baseline itself improved from 0.62→0.68 full@20 on a
  100q re-run purely from the banlist fix**, before any decompose logic was
  even involved. Against that *fresh* baseline, decompose's edge shrank to
  +0.10 full@10 (still significant), +0.06 full@20 (no longer significant).
  The earlier "+0.44" figure quoted mid-session was measuring against a
  *stale* pre-fix baseline — don't reuse it.
- Tried and **rejected** for retrieval: typed facet extraction (subject/org/
  place buckets), forcing a date/acronym word into the literal AND query
  (significantly *hurts* — rarity-based ranking lets a wrong guess jump the
  queue ahead of correct literal wording), labeled-field decomposition
  (statistical wash vs free-text), acronym-widening in 3 different forms
  (flat to negative every time, even after fixing a diagnosed ranking bug).
  See "Dead ends" for the specific failure traces — don't re-try these without
  a genuinely new angle.

**Selection stage** (exact = correct final pick; ceiling = right tables were
even visible):
- Production's actual mechanism = the lab's `inspect3` arm (cards + inline-
  inspect up to 3 candidates before deciding) — confirmed by reading
  `confirm_unified_selection` directly, not assumed.
- `cards` 0.46 exact / 0.76 ceiling, `inspect3` (=production) 0.50/0.76,
  `swarm20` (parallel per-candidate mini-agents) 0.50/0.82.
  **Swarm's real finding: main-agent context tokens 4,880→1,086 (77% less)
  for identical exact accuracy** — proves context can be cut hard with zero
  accuracy cost, but swarm's *independent* per-candidate verdicts never
  compare candidates against each other, so it doesn't fix exact accuracy.
- `distinguishing_prefilter` (deterministic, zero LLM cost, same-family
  sibling disambiguation via real inspected date/place vs the question):
  **+0.04 exact, significant [+0.01,+0.08]** on curated100, same direction
  not-significant on sel100, ceiling unchanged both times. The one
  selection-stage change all session with real significance behind it.
- Adding a comparative LLM call on top of the prefilter: no significant
  marginal gain either set.

## Bugs found and fixed this session (beyond the original banlist work)

1. **`search_keyword_concepts` wasn't actually splitting multi-word concepts**
   before the banlist comparison — fixed via `split_keywords()` in
   `keyword_terms.py` (this was the original session-opening bug report).
2. **`distinguishing_prefilter`'s phrase-matching narrowed on the first
   matching phrase and stopped**, instead of accumulating matches across all
   phrases the question states (the year-based path already did this
   correctly via set intersection). This meant a question needing two
   same-family siblings distinguished by two different stated phrases (e.g.
   two different areas) would silently drop the second one. Fixed in
   `experiments/retrieval_lab/arms_select2.py`; reproduced against the exact
   pre-fix logic to confirm the bug was real, then verified the fix in
   `tests/test_distinguishing_prefilter.py` (6 tests).
   - **Caveat**: neither benchmark set used to validate the prefilter
     (curated100, sel100) contains *any* question needing 2+ gold tables from
     the same family (checked directly via `family_of()`), so this scenario
     has no real-world benchmark coverage yet — the fix is proven correct
     against a realistic controlled fixture, not against a live regression
     number.

## Dead ends — tried, measured, rejected (don't redo without a new angle)

- **Typed facet extraction** (subject/org/place/time buckets): mild positive
  without time, materially *worse* when time was included as a forceable AND
  word.
- **Forcing a guessed/inferred word into the literal AND query** (date,
  acronym expansion): consistently backfires. Root cause found precisely:
  `candidate_subsets()` ranks by raw rarity (IDF), and a wrong guess is often
  *maximally rare* — which makes rarity-ranking actively reward it. Tried
  fixing the ranking (rank literal words ahead of guessed ones) — partially
  recovered the specific regressions it caused, but a *different* set of
  regressions appeared (guessed words still compete for the "pick 1-3" slots
  even when they can no longer win on ranking). Net: still not better than
  baseline after the ranking fix.
- **Labeled-field decomposition** (subject/date/edition/agency/place as
  separate JSON fields instead of one free-text phrase): statistical wash.
  Concrete failure modes: collapsed two distinct dates in one clause into a
  single table request (needed two), and confidently expanded an agency
  acronym to its official full name when the real portal `publisher` field
  used a *different* organisation's name (the acronym's actual parent
  department) — the "improvement" only worked when the guessed expansion
  happened to match the real metadata, which isn't reliable.
- **Aggregation-word stoplist + decompose prompt patch, acronym mechanism
  removed**: safe (zero harm across two independent 50q/100q sets) and
  slightly cheaper, but not a measurable accuracy win either. Worth keeping
  as a free efficiency tweak, not worth promoting on its own.

## Proposed selection prototype — designed, not yet built

4 stages, each reusing a proven-or-already-existing mechanism, explicitly
designed to be retrieval-method-agnostic (operates on the candidate table-ID
pool regardless of whether keyword/duckdb/semantic/hybrid produced it) and
annotated by how much each depends on model strength (relevant for the
`gpt-oss-20b` test):

0. **Deterministic same-family prefilter** (zero model dependency, already
   proven, bug-fixed this session) — run first on whatever pool retrieval
   produced.
1. **Parallel per-candidate briefing** (the swarm mechanism, but over the
   *whole* surviving pool, not capped at 3) — one cheap parallel call per
   candidate → compact verdict + supplies/lacks + `join_keys` (the schema
   already asks for this, just unused downstream today). Narrow single-focus
   task, the best candidate for testing with `gpt-oss-20b`.
2. **Parallel join/union precompute** (zero model dependency — reuses the
   existing pure-Python `check_join_union`, just runs it proactively on
   plausible pairs instead of waiting for the agent to request it one pair at
   a time via its own tool calls).
3. **Single listwise comparative pick** — the one genuinely untested,
   genuinely-needs-reasoning stage. Sees all of stage 1's compact verdicts +
   stage 2's join/union table together, has to compare candidates against
   each other (the thing swarm's independent scoring couldn't do) and commit
   to a final selection. Open question whether `gpt-oss-20b` holds up here
   given the much-smaller, pre-cleaned input, or whether it needs 120b.

Proposed test matrix (not run yet):

| run | Stage 1 model | Stage 3 model | tests |
|---|---|---|---|
| baseline | gpt-oss-120b | gpt-oss-120b | ceiling of the design |
| A | gpt-oss-20b | gpt-oss-120b | does narrow per-candidate judgment tolerate a small model? |
| B | gpt-oss-20b | gpt-oss-20b | does the cleaned-up input let the final decision run small too? |

## Immediate next steps (pick up here)

1. Build stage 0+1 as a new lab arm (`experiments/retrieval_lab/arms_select2.py`
   or a new file), confirm it reproduces the existing swarm token-savings
   numbers as a sanity check before adding anything new.
2. Add stages 2+3, full pipeline, gpt-oss-120b throughout — get the baseline
   row of the test matrix.
3. Re-run stages 1 and 3 with `gpt-oss-20b` substituted (runs A and B) —
   same question set, same everything else.
4. Not started: semantic-only and duckdb_agentic retrieval pools have never
   been tested at the selection stage in this lab (only keyword/hybrid/
   decompose pools have) — needed before the "4 retrieval methods" test
   matrix the user wants can actually run end to end.

## Where things live

- Production pipeline: `src/lakegen/agent_tools/tools_p12.py` (search +
  selection tools), `src/lakegen/experiment_config.py` (`DiscoveryConfig`),
  `src/lakegen/core/resources.py` (`get_llm`, model registration).
- Lab: `experiments/retrieval_lab/` — `arms_retrieval.py` (retrieval arms),
  `arms_select.py` + `arms_select2.py` (selection arms), `common.py` (shared
  plumbing: `LLM` class with disk cache, `KeywordSession` wrapping the real
  production tool, `equivalents()`/copy-aware scoring), `orqa10.py` (question
  sets), `report_stats.py` (paired bootstrap stats), `run_lab.py` (CLI runner).
  Rename history/naming convention: `rename_map.py`.
- Question sets used: `benchmark/50q_uk_unseen_20260918.json` (primary
  curated set, 33 single/17 multi), `benchmark/100q_uk.json`,
  `experiments/retrieval_lab/cache/curated100_raw.json` (built this session:
  the 84-question curated set + 16 more, 51 multi/49 single — built because
  every OrQa-sourced set had only ~2 multi-table questions), OrQa-sourced sets
  in `cache/orqa100_raw.json` / `orqafull_raw.json` (from
  `~/gits/orqa/data/uk/candidates_discovery/generated_queries_semantic.json`,
  which regenerates live — these are frozen snapshots).
- Tests: `tests/test_ban_shrinking.py`, `tests/test_keyword_terms.py`,
  `tests/test_decompose_preview_search.py`,
  `tests/test_distinguishing_prefilter.py` are the ones from this session.
  Full existing suite (~212 relevant tests) passes clean; a separate ~30
  failures across `test_delegated_selection.py`, `test_join_keys.py` (1),
  `test_keyword_memory.py`, `test_parallel_inspection.py`,
  `test_requirement_ledger.py` are pre-existing and unrelated (confirmed via
  `git stash` comparison against pre-session code) — don't spend time on them
  unless specifically asked to.

## Known unresolved caveats

- `uk` Solr core still has generic "CSV" titles for ~40% of gold tables
  (title=resource name bug); `uk_fixed` (built on branch `testing-grounds`)
  has correct titles but the retrieval lab's `common.py` still hardcodes
  `LocalSolrClient("uk")` — switching and re-running would be a real,
  untested variable that could move every retrieval number reported.
- Synonyms fix for licence/license Snowball-stemming mismatch: identified,
  can't apply (no filesystem access to the Solr container).
- The H0 (production hybrid) pool-size confound from early selection-stage
  testing (baseline used top-10, augmented arms used top-20) was flagged
  repeatedly but never cleanly resolved with a same-pool-width rerun.
