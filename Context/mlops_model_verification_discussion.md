# Model Verification in Actuarial MLOps: Discussion & Design

## Status: Active Design Document

**Date**: 2026-02-21
**Participants**: Kevin (Actuarial/Engineering Lead), Claude (Architecture)

---

## 1. Problem Statement

Support engineers are receiving requests from end users (underwriters) to "check the actual calculations of a model." The current workflow involves:

- GLM coefficients extracted into a **manual lookup table**
- Support engineers take the input data and perform a **dot product by hand** to verify the model output
- This is time-consuming, fragile, and fundamentally doesn't scale

The immediate trigger is usually an underwriter saying: *"I priced this earlier this week with what I believe is the same data and now I'm getting different results."*

### Why this matters now

The team is transitioning from a manual process (fit GLM, extract coefficients, build lookup table) to **containerized model inference**. This is the right moment to design the verification/debugging workflow into the system rather than bolting it on later. Additionally, as models grow more complex (GLMMs, GBMs, nonlinear models), the lookup-table approach becomes impossible.

---

## 2. Clarifying Questions & Answers

These questions shaped the architecture significantly.

### Q1: Who is the end user?

**Answer**: Underwriters pricing individual risks in the commercial/surplus lines market.

**Impact**: The typical scenario is not "explain the math" but rather "I got a different number and I don't know why." This reframed the entire problem from **calculation verification** to **prediction diffing**.

### Q2: What's the volume of these requests?

**Answer**: Several per month. Not a constant stream.

**Impact**: We don't need a full self-service portal. Investment should go into **support tooling** (diagnostic endpoints, audit trail) rather than end-user-facing explainability dashboards. The tooling needs to make each investigation fast, not handle thousands of concurrent investigations.

### Q3: Is there a regulatory driver?

**Answer**: Commercial/surplus lines, so less regulatory burden than personal lines. However, there could be questions from regulators, so the capability should exist.

**Impact**: We should build the audit trail and explainability infrastructure, but it doesn't need to meet the stringent per-factor-decomposition requirements of personal lines filings. The priority is operational support, with regulatory readiness as a secondary benefit.

### Q4: What does the container expose?

**Answer**: Currently planning a `/predict` endpoint, but we have full control to build whatever endpoints we want.

**Impact**: This is a greenfield opportunity. We can design the API contract to include diagnostic endpoints from day one rather than retrofitting them.

### Q5: How are models deployed today?

**Answer**: Very manual process: fit a linear model, extract coefficients, build lookup table. Moving to containerized model inference as the new standard.

**Impact**: The architecture needs to replace the lookup table entirely, not augment it. The container becomes the single source of truth for both predictions and explanations.

### Q6: Does the underwriter's system store prediction references?

**Answer**: The system stores some account information but relies heavily on intermediate calculations logged to a database. No prediction ID concept exists today, but there's no reason it couldn't be added for new greenfield models.

**Impact**: The prediction ID is a foundational element. Every prediction gets a UUID that the calling system stores. When the underwriter calls support, they reference this ID (or support looks it up by policy + timestamp). This enables the entire `/compare` workflow.

### Q7: Where does feature enrichment happen?

**Answer**: The container receives a fully-formed JSON payload. There will be some additional feature engineering inside the container for the model sequence, but no external table lookups needed.

**Impact**: Simplifies the logging architecture. We need to capture (a) the raw inbound JSON and (b) the derived features after internal engineering. We don't need to worry about capturing the state of external reference tables at prediction time.

### Q8: Language preference?

**Answer**: First implementation will be R, but the team is polyglot (R, Python, potentially others). The architecture must be language-agnostic.

**Impact**: The prediction event schema and API contract must be language-neutral (JSON-based). Individual containers can be implemented in whatever language suits the model. The logging infrastructure should not depend on any particular language runtime.

---

## 3. The Core Reframe

The initial framing was: **"How do we let support engineers check model calculations without a lookup table?"**

Through the Q&A, this reframed to: **"The underwriter says the data is the same. It almost never is. How do we prove what changed, quickly?"**

This is fundamentally a **diff problem**, not a calculation verification problem. The underwriter doesn't need to see coefficients or matrix algebra. They need to know:

> "Last time territory was 001, this time it's 003. That accounts for the $200 difference in premium."

This reframe drives the entire architecture toward **prediction logging + comparison** rather than **coefficient extraction + manual verification**.

---

## 4. Solution Space Explored

We evaluated six approaches before settling on a recommended architecture.

### 4.1 Enriched Prediction API ("the model explains itself")

Every `/predict` response includes not just the prediction but a full decomposition: base rate, factor contributions, confidence interval, data quality flags. For GLMs this is exact (multiplicative factors). For complex models, use SHAP values or similar attribution.

**Verdict**: Essential. Replaces the lookup table. Becomes part of the core `/predict` response or the `/explain` endpoint.

### 4.2 Prediction Audit Trail ("every prediction is reproducible")

Log every prediction with full input/output/metadata to a database. Support can look up any historical prediction and replay it.

**Verdict**: Essential. This is the foundation that enables `/compare` and `/history`. Low implementation cost, enormous value.

### 4.3 Support Sandbox / Shadow Container (Kevin's original idea)

A non-production replica where support can submit test predictions, run what-if scenarios, and compare model versions.

**Verdict**: Useful but secondary. The core "why did my price change" problem is better solved by the prediction ledger + `/compare`. The sandbox adds value for what-if analysis and pre-deployment testing. Could be implemented as a flag on the container (`mode=sandbox`, predictions not logged to production ledger) rather than a separate deployment.

### 4.4 Automated Guardrails ("catch problems before users do")

The serving layer proactively flags unusual predictions: out-of-distribution inputs, extreme tail predictions, rare factor combinations, prediction instability.

**Verdict**: High value, medium effort. Reduces the volume of support requests by catching the surprising cases before they reach the underwriter. Implemented as quality flags in the prediction event.

### 4.5 Interactive Model Explorer ("the digital rate manual")

A web tool auto-generated from the model artifact showing factor relationships, partial dependence plots, and interactive input manipulation.

**Verdict**: Nice to have. Not needed for the immediate support problem but valuable for stakeholder trust and onboarding. Could be a future phase.

### 4.6 Surrogate Model Governance ("trust but verify")

Run a simplified interpretable model alongside the complex model. Flag significant disagreements.

**Verdict**: Situational. Only relevant when deploying very complex models (deep learning, large ensembles) where the primary model is fundamentally opaque. Not needed for GLMs/GLMMs.

### Prioritization

| Priority | Solution | Effort | Value |
|----------|----------|--------|-------|
| **Phase 1** | Prediction Audit Trail (#4.2) | Low | Foundation for everything else |
| **Phase 1** | Enriched Prediction API (#4.1) | Medium | Replaces lookup table |
| **Phase 2** | Automated Guardrails (#4.4) | Medium | Reduces support ticket volume |
| **Phase 2** | Support Sandbox (#4.3) | Medium | What-if analysis |
| **Phase 3** | Model Explorer (#4.5) | High | Stakeholder trust |
| **As needed** | Surrogate Model (#4.6) | High | Complex model governance |

---

## 5. Recommended Architecture

### 5.1 Container Endpoints

The model container exposes four endpoints:

| Endpoint | Purpose | Query Source |
|----------|---------|-------------|
| `POST /predict` | Score a risk, return prediction + prediction_id | Underwriter's system (production) |
| `GET /explain/{prediction_id}` | Full decomposition of a logged prediction | Support engineer |
| `POST /compare` | Diff two predictions (inputs + attributed output change) | Support engineer |
| `GET /history?policy_id=X&days=N` | All predictions for a given risk | Support engineer |

### 5.2 The Prediction Event Schema

Every `/predict` call generates a structured prediction event (language-agnostic JSON):

```json
{
  "prediction_id": "uuid",
  "timestamp": "ISO8601",
  "model_id": "workcomp-freq-v2.3",
  "model_artifact_hash": "sha256:abc123...",

  "raw_input": { "...the full inbound JSON..." },
  "caller_id": "underwriter-session-xyz",

  "pipeline_steps": [
    {
      "step_name": "feature_engineering",
      "input_features": {},
      "output_features": {},
      "duration_ms": 12
    },
    {
      "step_name": "frequency_model",
      "prediction": 0.34,
      "link_scale_prediction": -1.08,
      "decomposition": {
        "intercept": -1.20,
        "territory_003": 0.45,
        "class_8810": 0.15
      },
      "duration_ms": 8
    },
    {
      "step_name": "severity_model",
      "prediction": 28500,
      "decomposition": {},
      "duration_ms": 6
    },
    {
      "step_name": "pure_premium",
      "prediction": 9690,
      "formula": "frequency * severity",
      "duration_ms": 1
    }
  ],

  "final_prediction": 9690,
  "prediction_interval": [6200, 14800],

  "warnings": ["class_code_rare_in_training"],
  "ood_score": 0.12,
  "total_duration_ms": 27
}
```

### 5.3 Dual-Destination Storage

| Store | Purpose | Retention | Powers |
|-------|---------|-----------|--------|
| **Postgres** | Operational queries | 90-180 days | `/compare`, `/history`, `/explain` |
| **Snowflake** | Long-term audit + analytics | Years | Drift monitoring, regulatory, model performance |

### 5.4 Logging Patterns (Language-Agnostic)

**Option A: Direct async write** -- Container writes to Postgres asynchronously after returning the response. Simplest. Fine for current volume.

**Option B: Structured log + shipper** -- Container emits JSON to stdout. A log shipper (Fluentd/Vector) routes to both Postgres and Snowflake. Cleanest separation of concerns. Truly language-agnostic since the model code never touches a database driver.

**Recommended**: Start with Option A for simplicity, design the prediction event format so that migrating to Option B later is just a configuration change (the event schema stays the same).

### 5.5 The /compare Workflow (the core support use case)

```
Support engineer receives call:
"I priced policy X on Monday, got $4200. Today it's $4650. Same data."

1. Look up predictions:
   GET /history?policy_id=X&days=7
   → Returns: pred_abc123 (Monday, $4200), pred_def456 (Thursday, $4650)

2. Compare them:
   POST /compare { "prediction_a": "abc123", "prediction_b": "def456" }

3. Response:
   {
     "input_diff": {
       "territory": {"was": "001", "now": "003"},
       "all_other_fields": "identical"
     },
     "model_version_changed": false,
     "output_diff": {"was": 4200, "now": 4650, "delta": 450},
     "attribution": {
       "territory_change": +450
     }
   }

4. Support tells underwriter:
   "Territory changed from 001 to 003 between submissions.
    That accounts for the full $450 difference."
```

Time to resolution: **minutes instead of hours**.

---

## 6. Proactive Monitoring (Built on the Prediction Ledger)

Once prediction events flow into Snowflake, lightweight SQL analytics can detect issues before underwriters notice:

- **Prediction stability**: Same risk scored multiple times with divergent results (> 20% swing)
- **Data drift**: Feature distributions shifting from training baseline (z-score > 2)
- **Volume anomalies**: Sudden drops/spikes in prediction volume
- **OOD frequency**: Increasing rate of out-of-distribution flags
- **Model version impact**: Distribution shift after model deployment

---

## 7. Open Questions & Next Steps

### Decided
- Container will expose `/predict`, `/explain`, `/compare`, `/history`
- Prediction events logged with full pipeline trace
- Dual-destination: Postgres (operational) + Snowflake (audit/analytics)
- First implementation in R (plumber), architecture is language-agnostic
- Prediction IDs returned to caller and stored by underwriting system

### To Design Next
- [ ] R/plumber implementation skeleton (middleware pattern for instrumentation)
- [ ] Postgres schema for prediction events (JSONB-heavy for flexibility)
- [ ] Snowflake ingestion pattern (batch vs. Snowpipe)
- [ ] `/compare` attribution logic (exact for GLMs, SHAP for complex models)
- [ ] How the sandbox/what-if mode works alongside production
- [ ] OOD detection approach (what metric, what threshold)
- [ ] Integration pattern with existing underwriting systems (how prediction_id flows back)

### Open Questions
- What authentication/authorization model for the diagnostic endpoints? (Support-only vs. broader access)
- Should `/explain` be computed on-the-fly or stored at prediction time? (Storage vs. compute tradeoff)
- How do we handle model version transitions in `/compare`? (Attribution across different model structures)
- What's the deployment target? (Kubernetes, ECS, bare Docker, cloud-specific?)

---

## 8. Architecture Review: Critical Evaluation

*Independent review performed by a subagent acting as a senior MLOps architect and actuarial systems engineer. The goal was to stress-test the design before proceeding to implementation.*

### High Severity

#### 8.1 Async write failure creates silent audit trail gaps

The Phase 1 recommendation (Option A: direct async write) has a critical failure mode. If the Postgres write fails after the response has already been returned to the caller, the prediction event is lost permanently. There is no fallback — no stdout log (that's Option B), no message queue (that's Option C), and the container's memory is volatile. At several support requests per month, losing even one event means a support case that cannot be resolved.

**Recommendation**: At minimum, write the prediction event to a durable local store (even a file on a mounted volume) *before* attempting the async DB write. This gives you a recovery path. Alternatively, adopt Option B (structured log to stdout + shipper) from day one — it's only marginally more complex and eliminates this failure mode entirely.

#### 8.2 In-memory write buffer is not durable

Diagram 5 in the architecture doc shows a `Write Buffer (in-memory queue)` feeding an `Async DB Writer`. In a containerized environment, this buffer is lost on any container crash, OOM kill, or restart. This compounds the issue in 8.1 — the window of data loss is wider than a single failed write; it includes any events sitting in the buffer at the time of a crash.

**Recommendation**: Replace the in-memory buffer with a file-based buffer on a mounted volume, or commit to writing to stdout as the primary log path. The pattern of "structured JSON to stdout → Docker captures it → shipper routes it" is standard in containerized deployments and is durable by default.

#### 8.3 Cross-version `/compare` is the hardest unsolved problem

Model deployments happen precisely when prices are expected to change, which is precisely when underwriters call support. The current design acknowledges this as an "open question" but it's actually the most common real-world scenario for the `/compare` endpoint. When inputs changed AND the model version changed simultaneously, the attribution logic has no way to decompose "how much is from the input change vs. how much is from the model change" without running the counterfactual (old inputs through new model, or new inputs through old model).

**Recommendation**: At minimum, commit to a degraded-but-honest response format for cross-version comparisons: *"Model version changed from v2.3 to v2.4. Input differences account for approximately $X of the $450 delta. The remaining $Y is attributable to the model version change. Granular cross-version attribution is not available."* The approximate input attribution can be computed by running the changed inputs through whichever model version is currently loaded. Consider keeping the previous model version loadable (as a secondary artifact) specifically for this use case.

#### 8.4 No auth/access control design for sensitive endpoints

The prediction events contain full input payloads (potentially including business names, addresses, employee counts — commercially sensitive data) and full model decompositions (which reveal the proprietary rating algorithm). The diagnostic endpoints (`/explain`, `/compare`, `/history`) expose this information. There is no discussion of who should have access: should underwriters see decompositions? Should all support engineers see all accounts? Can a support engineer query any policy, or only those assigned to them?

**Recommendation**: Define role-based access before Phase 1 ships. At minimum: (a) `/predict` is accessible to the underwriting system via API key, (b) diagnostic endpoints require a separate support-role credential, (c) `/history` results are filtered by the caller's authorization scope. Consider whether diagnostic endpoints should be on a separate internal-only service rather than the same container that serves production predictions.

### Medium Severity

#### 8.5 `policy_id` is load-bearing but undefined

The `/history` endpoint depends on `policy_id` to find all predictions for a given risk. The ERD shows it as a column on `PREDICTION_EVENT`. But the Q&A establishes that the container receives a "fully-formed JSON payload" — it's never specified whether `policy_id` is a field in that JSON, a request header, or something derived from `caller_id`. This is a small gap but it's foundational: if the calling system doesn't provide a consistent `policy_id`, the entire `/history` use case breaks.

**Recommendation**: Explicitly define `policy_id` as a required field in the `/predict` request schema. Document what it should contain (quote number, submission ID, policy number — these are different things in insurance). Consider whether you also need a `submission_id` or `account_id` as a higher-level grouping.

#### 8.6 Sandbox flag should be an env var, not a per-request parameter

The architecture suggests the sandbox could be "the same image with a sandbox flag." If that flag is a per-request parameter or header, a misconfigured production integration could accidentally pass `sandbox=true`, causing real underwriting predictions to silently bypass the audit trail. This is a quiet, hard-to-detect failure mode.

**Recommendation**: The sandbox flag must be an environment variable set at container startup, not a per-request parameter. A sandbox container is a sandbox from the moment it starts; it cannot be toggled per-request.

#### 8.7 Snowflake not justified at stated volume

Several support requests per month implies dozens to low hundreds of predictions per day for a commercial lines book. This is trivially small for a single Postgres instance. The monitoring queries listed in Section 6 (drift detection, prediction stability, volume anomalies) are simple aggregations that Postgres handles easily at this scale for years. Snowflake adds: compute credit costs, a second schema to maintain, an ETL/Snowpipe pipeline to build and monitor, and a second failure mode. None of the listed analytics require a columnar warehouse at this volume.

**Recommendation**: Defer Snowflake entirely. Use Postgres as the single store with a sensible retention policy (keep everything — at this volume, storage is negligible). Revisit Snowflake only when (a) prediction volume grows by 10-100x, (b) you need to join prediction data with other enterprise data already in Snowflake, or (c) you need query patterns that genuinely benefit from columnar storage. Re-frame the current Snowflake recommendation as a "future scaling path" rather than a Phase 2 deliverable.

#### 8.8 OOD score is in the schema with no definition

The prediction event schema includes `ood_score: 0.12` as a field, and the ERD has it as a column. But there is no specification of how it's computed, what model or method produces it, what the score means (probability? distance? percentile?), or how it was calibrated. OOD detection for tabular insurance data is non-trivial — common approaches (isolation forest, Mahalanobis distance, kernel density) each require their own training artifacts, versioning, and validation.

**Recommendation**: Remove `ood_score` from the Phase 1 schema. Replace it with a more honest `warnings` array that captures simple, rule-based checks: "feature X outside training range [min, max]", "categorical level Y not seen in training data." These are trivially computable, universally understood, and don't require a separate model. Introduce a proper OOD score in Phase 2 when you've had time to evaluate and validate an approach.

#### 8.9 Policy lifecycle / transaction type not modeled

The schema treats every prediction as an independent event linked only by `policy_id`. But commercial insurance policies have a lifecycle: quote, bind, endorsement, renewal, audit, cancellation. A re-rating at endorsement is fundamentally different from a new business quote, but both appear identically in the prediction ledger. When support pulls `/history?policy_id=X`, they see all transaction types mixed together. Comparing a quote to an endorsement re-rating produces diffs full of expected changes (exposure updated, deductible changed, etc.) that are noise, not signal.

**Recommendation**: Add a `transaction_type` field to the prediction request schema (enum: `quote`, `bind`, `endorsement`, `renewal`, `audit`, `other`). This allows `/history` to be filtered by transaction type and `/compare` to flag when the two predictions being compared are from different lifecycle stages: *"Note: Prediction A was a new business quote; Prediction B is an endorsement re-rating. Exposure and deductible changes are expected."*

#### 8.10 Schema evolution across model versions not addressed

When `model_id` changes from `workcomp-freq-v2.3` to `workcomp-freq-v3.0`, the decomposition keys may change: `territory_003` might become `territory_zone_B`, new features may be added, old features may be dropped. The `/compare` endpoint will produce diffs showing every decomposition key as changed, even if the underlying rating factor is the same concept renamed. There is no model manifest that maps feature names across versions, and no namespacing convention for decomposition keys.

**Recommendation**: Require each model version to ship with a manifest file that declares its feature names and maps them to canonical factor names. For example, both `territory_003` (v2) and `territory_zone_B` (v3) map to canonical factor `territory`. The `/compare` endpoint uses canonical names when diffing across versions. This is a metadata discipline, not a code change — but it needs to be established as a convention before the first model ships.

### Notable Insights

#### 8.11 The "same data" problem likely originates before the container

The architecture captures raw input and derived features from inside the container. But the most common source of "I submitted the same data" disputes in commercial lines is feature enrichment that happens *upstream* of the container. Example: the underwriter enters the same street address, but a geocoding service maps it to a different territory code between Monday and Thursday. The container receives a different `territory` value both times — and correctly reports the diff — but the underwriter is right that *they* didn't change anything.

This is outside the container's control, but the architecture should acknowledge this explicitly. The `/compare` response could include guidance: *"Territory changed from 001 to 003. If the underwriter reports submitting the same address, this may indicate an upstream geocoding or enrichment change — investigate the quoting system's territory assignment logic."* More practically, if the raw address (or other pre-enrichment fields) can be included in the JSON payload alongside the enriched values, the container can log both and `/compare` can distinguish "underwriter changed the input" from "enrichment changed the output."

#### 8.12 The 80% solution we didn't seriously consider

A structured JSON prediction log written to a flat file or S3, combined with a simple `diff` tool (or a 20-line script), would solve the stated problem for 18-24 months. Support pulls two log entries by timestamp and policy, runs a JSON diff, and reports back. No Postgres schema, no async writer, no `/compare` endpoint, no dual-destination storage. Just structured logs and a diff.

This is not the right long-term answer, but it is worth being explicit that the team is choosing to build infrastructure *now* because they expect model complexity and volume to grow — not because the immediate problem at several requests per month requires it. This is a legitimate investment decision, but the discussion document should frame it as such. If the team is resource-constrained, the 80% solution buys significant time at near-zero cost, and the fuller architecture can be built incrementally on top of it (the prediction event schema is the same either way — only the storage and query layers differ).

#### 8.13 Diagnostic endpoints should be separated from the prediction serving container

The dominant ML monitoring platforms (Evidently, Arize, WhyLabs, Fiddler) all separate the logging/analysis concern from the model serving concern. They operate as sidecars or separate services. The current architecture puts the Postgres reader (for `/compare`, `/explain`, `/history`) inside the same container that serves `/predict`. This means diagnostic queries compete for CPU, memory, and network with production prediction traffic.

At current volume this is a non-issue. But it sets an architectural pattern that becomes painful to unwind later — especially if support starts running complex `/compare` queries during peak quoting hours. The cleaner pattern is: the model container does exactly two things (score and emit events), and a separate lightweight service owns the diagnostic endpoints by reading from Postgres directly. This also cleanly solves the auth separation problem (8.4): the production `/predict` API and the internal diagnostic API have different access control, different scaling profiles, and different SLAs.
