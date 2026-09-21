# Reviewed coder pipeline: telemetry and staged rollout

This documents the operational side of `lakegen.reviewed_coder` (the plan-
fidelity judge, deterministic validator, and post-execution code judge added
around the existing `phase3_generate_and_execute`). The mechanism itself,
its architecture, and why it exists are documented in
[`src/lakegen/reviewed_coder.py`](../src/lakegen/reviewed_coder.py)'s module
docstring. This file covers: how to turn it on, what it costs, what to watch
before trusting it, and the order in which to turn it on for real traffic.

## Enabling it

Off by default. Two independent flags in `ExperimentConfig.reviewers`:

```yaml
reviewers:
  plan: true   # plan/lineage judge (question-fidelity only)
  code: true   # post-execution code judge (gold-free)
  stage_max_retries: 3  # per-stage budget; validated at 3, tunable 1-5
```

The deterministic query validator (value-grounding + date-parse-sanity, the
direct fix for the motivating fiscal-year-span bug) runs whenever either flag
is on -- it's cheap and doesn't need its own switch. Turning on `plan` or
`code` alone is meaningful and independently useful; they don't require each
other.

When either flag is on, the reviewed pipeline replaces the existing
`MAX_CODE_ATTEMPTS`-bounded outer retry loop in `service.py` and
`ui/workflow.py` for that question -- the outer loop runs exactly once,
since the reviewed pipeline owns its own bounded internal retrying per
stage. See the "cost" section below for what that internal retrying
actually spends.

## What "validated" means today, and what it doesn't yet

The design (three independently-bounded stage loops: plan/lineage →
validator → code judge, never confidently returning an answer that was
never approved) was built and validated across **four rounds of live
100-question testing against one real UK portal**, entirely in a scratchpad
harness, reusing the real `phase3_generate_and_execute` for every code
generation call. The measured numbers from that final round:

- 60/100 questions reached a fully-approved, trustworthy answer; 77% of
  those matched the existing (non-reviewed) pipeline's answer exactly.
- The other 40/100 correctly declined with a stated reason instead of
  returning an unapproved guess.
- Average 2.17 real code-generation calls per question, max 5 observed
  (never the theoretical worst case of 9 = 3 stages × 3 tries).
- The plan-judge stage is the dominant bottleneck (116 rejections vs. 70
  approvals across all attempts); the validator (2 rejections) and code
  judge (39 rejections) are comparatively easy to satisfy once a plan is
  approved.
- Two real, concrete bugs were caught and fixed during that validation (a
  malformed-header table silently mis-parsing a date column to 100% NaT; a
  `.eq()`-method-call filter the AST literal-comparison collector originally
  missed) -- both are now covered by regression tests
  (`tests/test_value_grounding.py`).

**What this does NOT yet establish**: whether the ~60/40 split and the ~2.2
average generation-call cost generalize beyond that one 100-question sample
on one portal. Before changing any default, re-check both numbers on a
larger, more representative question set and a different portal using the
telemetry below -- that is the explicit purpose of Phase 4's telemetry
work, not a formality.

## Telemetry

Every `QueryResult` now carries (`src/lakegen/service_models.py`):

| Field | Meaning |
|---|---|
| `review_pipeline_used` | `True` iff the reviewed pipeline ran for this question (i.e. `reviewers.plan` or `.code` was on) |
| `review_outcome` | `"validated"` or `"declined"`, taken from the pipeline's own closing summary event |
| `review_stage_attempts` | e.g. `{"plan": 2, "validator": 1, "code_judge": 1}` -- how many tries each stage actually took |
| `review_generation_calls` | total real `phase3_generate_and_execute` calls spent on this question |
| `review_trace` | the full per-stage-attempt trace (lineage violations, judge rationale/feedback, validator grounding violations) -- everything needed to reconstruct why a question was declined or how many retries it took |

`_record_review_telemetry` (`src/lakegen/service.py`) populates these
unconditionally whenever the reviewed pipeline ran; it is a pure recording
step with no effect on the question's outcome. A plain (non-reviewed) run
leaves all of these at their defaults (`review_pipeline_used=False`,
`review_trace=[]`).

Before promoting any default, pull `review_outcome`, `review_stage_attempts`
and `review_generation_calls` across a real traffic sample and compare
against the numbers above. A materially higher decline rate or generation-
call cost on real traffic is a signal to revisit `stage_max_retries` or the
judge prompts, not to change the default anyway.

## Staged rollout order

1. **`reviewers.plan` alone**, in a test/staging experiment config first. It
   is the dominant, most bug-motivated stage (the fiscal-year-span bug and
   the two regressions caught during validation were both plan/validator
   issues, not code-judge issues) and has zero dependency on the code judge.
   Watch `review_outcome`/`review_stage_attempts` against a broader question
   set than the 100-question sample before trusting the 60/40 split
   generalizes.
2. **Add `reviewers.code`** once the plan-stage decline rate on real traffic
   looks acceptable.
3. **Only consider flipping either default to `True`** after that -- and
   even then, pair it with a UX decision about how "I can't confidently
   answer this, here's why" (`rejected_reason` on a `finalization_mode ==
   "review_declined"` result) is surfaced to the end user. That UX decision
   is outside this pipeline's scope; flag it to whoever owns the
   user-facing surface before flipping a default that will show up as a
   real decline rate to real users.

## Explicitly out of scope (unchanged from the implementation plan)

- A genuine pre-code, LLM-authored structured plan with ordered
  steps/`produces` declarations (`coder_brief`/`AnalysisContractSchema` is
  confirmed unreliable in production -- frequently an almost-empty
  `"runtime_fallback"`; the plan/lineage stages judge `operation_trace` +
  the actual generated code post-hoc instead). Building a real planner is
  valuable, separate follow-on work.
- Re-verifying an earlier stage after a later stage triggers regeneration
  (documented, accepted tradeoff of `phase3_generate_and_execute` being an
  atomic generate+execute call).
- Sandbox/security hardening.
- `reviewers.dataset`/`.result`, `gates.plan`/`.result` -- still rejected by
  `ExperimentConfig.validate_supported_configuration`.
