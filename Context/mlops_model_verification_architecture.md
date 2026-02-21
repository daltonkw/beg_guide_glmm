# Model Verification Architecture: Diagrams

## Status: Active Design Document

**Date**: 2026-02-21
**Companion to**: `mlops_model_verification_discussion.md`

---

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph "Underwriting System"
        UW[Underwriter UI]
        QS[Quoting System]
    end

    subgraph "Model Container"
        API[API Layer<br/>plumber / FastAPI]
        FE[Feature Engineering]
        MOD[Model Sequence<br/>freq → sev → pure premium]
        INSTR[Instrumentation Layer<br/>logging + timing + decomposition]

        API --> FE --> MOD
        MOD --> INSTR
        INSTR --> API
    end

    subgraph "Support Tooling"
        STOOL[Support Interface<br/>CLI / simple web UI]
    end

    subgraph "Operational Store"
        PG[(Postgres<br/>90-180 day retention<br/>JSONB prediction events)]
    end

    subgraph "Analytical Store"
        SF[(Snowflake<br/>long-term retention<br/>audit + analytics)]
    end

    subgraph "Monitoring"
        DASH[Dashboards<br/>drift / volume / stability]
        ALERT[Alerts<br/>anomaly detection]
    end

    QS -->|"POST /predict<br/>JSON payload"| API
    API -->|"prediction +<br/>prediction_id"| QS
    QS -->|"stores prediction_id"| UW

    STOOL -->|"/compare<br/>/explain<br/>/history"| API
    API -->|"reads from"| PG

    INSTR -->|"async write"| PG
    PG -->|"batch ETL or<br/>log shipper"| SF
    SF --> DASH
    SF --> ALERT
```

---

## 2. Request Flow: `/predict`

```mermaid
sequenceDiagram
    participant UW as Underwriter System
    participant API as Container API
    participant FE as Feature Engineering
    participant FREQ as Frequency Model
    participant SEV as Severity Model
    participant AGG as Aggregation
    participant LOG as Logger
    participant PG as Postgres

    UW->>API: POST /predict {raw JSON}
    activate API

    API->>API: Generate prediction_id (UUID)
    API->>API: Record timestamp, caller_id

    API->>FE: Raw features
    activate FE
    FE->>FE: Derive/transform features
    FE-->>LOG: Step trace (input → output)
    FE->>FREQ: Engineered features
    deactivate FE

    activate FREQ
    FREQ->>FREQ: predict() + decompose()
    FREQ-->>LOG: Step trace (prediction + decomposition)
    FREQ->>SEV: Continue pipeline
    deactivate FREQ

    activate SEV
    SEV->>SEV: predict() + decompose()
    SEV-->>LOG: Step trace (prediction + decomposition)
    SEV->>AGG: Sub-model outputs
    deactivate SEV

    activate AGG
    AGG->>AGG: Combine (freq × sev)
    AGG-->>LOG: Step trace (final calculation)
    deactivate AGG

    API-->>UW: {prediction, prediction_id, warnings}

    LOG->>PG: Async write PredictionEvent
    deactivate API

    Note over PG: Full event stored:<br/>raw input, pipeline steps,<br/>decompositions, quality flags
```

---

## 3. Support Workflow: `/compare`

```mermaid
sequenceDiagram
    participant UW as Underwriter
    participant SUP as Support Engineer
    participant API as Container API
    participant PG as Postgres

    UW->>SUP: "I priced this Monday ($4200),<br/>now it's $4650. Same data."

    SUP->>API: GET /history?policy_id=X&days=7
    API->>PG: Query recent predictions
    PG-->>API: pred_abc123 (Mon, $4200)<br/>pred_def456 (Thu, $4650)
    API-->>SUP: Prediction list with IDs

    SUP->>API: POST /compare<br/>{pred_a: "abc123", pred_b: "def456"}
    API->>PG: Fetch both prediction events
    PG-->>API: Full events with inputs + pipeline traces

    API->>API: Diff raw inputs
    API->>API: Diff engineered features
    API->>API: Diff sub-model outputs
    API->>API: Compute attribution for each change

    API-->>SUP: {<br/>  input_diff: {territory: "001"→"003"},<br/>  model_version_changed: false,<br/>  output_diff: {delta: +$450},<br/>  attribution: {territory: +$450}<br/>}

    SUP->>UW: "Territory changed from 001 to 003<br/>between submissions. That's the<br/>full $450 difference."

    Note over SUP: Resolution time: minutes,<br/>not hours
```

---

## 4. Prediction Event Data Model

```mermaid
erDiagram
    PREDICTION_EVENT {
        uuid prediction_id PK
        timestamp created_at
        string model_id
        string model_artifact_hash
        jsonb raw_input
        string caller_id
        string policy_id
        numeric final_prediction
        numeric prediction_lower
        numeric prediction_upper
        numeric ood_score
        integer total_duration_ms
        jsonb warnings
    }

    PIPELINE_STEP {
        uuid step_id PK
        uuid prediction_id FK
        integer step_order
        string step_name
        jsonb input_features
        jsonb output_features
        jsonb decomposition
        numeric step_prediction
        numeric link_scale_prediction
        string formula
        integer duration_ms
    }

    PREDICTION_EVENT ||--o{ PIPELINE_STEP : "has steps"
```

---

## 5. Container Internal Architecture

```mermaid
graph LR
    subgraph "Container"
        subgraph "API Layer"
            EP_PRED["/predict"]
            EP_EXPL["/explain/{id}"]
            EP_COMP["/compare"]
            EP_HIST["/history"]
            EP_HEALTH["/health"]
        end

        subgraph "Instrumentation Middleware"
            TIMER[Timing]
            TRACER[Pipeline Tracer]
            DECOMP[Decomposition Engine<br/>GLM: exact coefficients<br/>Complex: SHAP values]
            QUALITY[Quality Checks<br/>OOD detection<br/>reasonableness bounds]
        end

        subgraph "Model Runtime"
            FE2[Feature Engineering<br/>R functions / Python transforms]
            M1[Model 1: Frequency]
            M2[Model 2: Severity]
            M3[Aggregation Logic]
            ART[Model Artifacts<br/>.rds / .pkl / .joblib]
        end

        subgraph "Persistence"
            BUF[Write Buffer<br/>in-memory queue]
            WRITER[Async DB Writer<br/>background process]
        end

        EP_PRED --> TIMER --> TRACER
        TRACER --> FE2 --> M1 --> M2 --> M3
        M1 --> DECOMP
        M2 --> DECOMP
        TRACER --> QUALITY
        QUALITY --> BUF --> WRITER

        EP_EXPL --> PG_READ[Postgres Read]
        EP_COMP --> PG_READ
        EP_HIST --> PG_READ
    end

    WRITER --> PG2[(Postgres)]
    PG_READ --> PG2
```

---

## 6. Dual-Destination Logging Architecture

```mermaid
graph TB
    subgraph "Model Container"
        PRED[/predict endpoint/]
        EVENT[PredictionEvent JSON]
        PRED --> EVENT
    end

    subgraph "Option A: Direct Write (Phase 1)"
        EVENT -->|"async write<br/>(R: callr::r_bg / promises)<br/>(Py: BackgroundTasks)"| PGA[(Postgres)]
    end

    subgraph "Option B: Log Shipper (Phase 2)"
        EVENT -->|"structured JSON<br/>to stdout"| DOCKER[Docker Log Driver]
        DOCKER --> SHIP[Log Shipper<br/>Fluentd / Vector]
        SHIP -->|"low-latency<br/>single row"| PGB[(Postgres)]
        SHIP -->|"batched<br/>bulk load"| SFB[(Snowflake)]
    end

    subgraph "Option C: Message Queue (Future)"
        EVENT --> MQ[Kafka / Redis Streams]
        MQ --> CONSUMER1[Consumer: Postgres Writer]
        MQ --> CONSUMER2[Consumer: Snowflake Loader]
        MQ --> CONSUMER3[Consumer: Alert Engine]
        CONSUMER1 --> PGC[(Postgres)]
        CONSUMER2 --> SFC[(Snowflake)]
    end

    style EVENT fill:#f9f,stroke:#333
```

---

## 7. `/compare` Attribution Logic

```mermaid
graph TB
    subgraph "Input: Two Prediction Events"
        A[Prediction A<br/>Monday, $4200]
        B[Prediction B<br/>Thursday, $4650]
    end

    subgraph "Step 1: Input Diff"
        DIFF[Compare raw_input fields<br/>field by field]
        A --> DIFF
        B --> DIFF
        DIFF --> CHANGED[Changed fields:<br/>territory: 001 → 003]
        DIFF --> SAME[Unchanged fields:<br/>class_code, exposure, ...]
    end

    subgraph "Step 2: Check Model Version"
        VCHECK{Same model<br/>version?}
        CHANGED --> VCHECK
        VCHECK -->|Yes| ATTR
        VCHECK -->|No| VDIFF[Model version also<br/>contributes to delta]
        VDIFF --> ATTR
    end

    subgraph "Step 3: Attribution"
        ATTR[Compute contribution<br/>of each changed input]

        GLM_PATH[GLM Path:<br/>Exact coefficient difference<br/>exp&#40;beta_new&#41; / exp&#40;beta_old&#41;]
        COMPLEX_PATH[Complex Model Path:<br/>SHAP value difference<br/>between the two predictions]

        ATTR --> GLM_PATH
        ATTR --> COMPLEX_PATH
    end

    subgraph "Output"
        RESULT["input_diff: {territory: 001→003}<br/>model_changed: false<br/>delta: +$450<br/>attribution: {territory: +$450}"]
    end

    GLM_PATH --> RESULT
    COMPLEX_PATH --> RESULT
```

---

## 8. Deployment Topology (Target State)

```mermaid
graph TB
    subgraph "Production"
        LB[Load Balancer]
        subgraph "Container Cluster"
            C1[Model Container<br/>Instance 1]
            C2[Model Container<br/>Instance 2]
        end
        LB --> C1
        LB --> C2
    end

    subgraph "Data Tier"
        PG[(Postgres<br/>prediction ledger<br/>90-180 day)]
        SF[(Snowflake<br/>full history<br/>analytics)]
    end

    subgraph "Support Access"
        STOOL[Support CLI / UI]
        STOOL -->|"diagnostic endpoints<br/>auth required"| LB
    end

    subgraph "Sandbox (Phase 2)"
        SANDBOX[Sandbox Container<br/>same image, sandbox flag<br/>predictions not logged<br/>to production ledger]
        STOOL -->|"what-if scenarios"| SANDBOX
    end

    subgraph "CI/CD Pipeline"
        CICD[Build Pipeline]
        CICD -->|"deploy same image"| C1
        CICD -->|"deploy same image"| C2
        CICD -->|"deploy same image"| SANDBOX
    end

    C1 --> PG
    C2 --> PG
    PG -->|"batch ETL"| SF

    subgraph "Monitoring"
        DASH[Dashboards]
        SF --> DASH
    end
```

---

## 9. Phase Roadmap

```mermaid
gantt
    title Implementation Phases
    dateFormat YYYY-MM-DD
    axisFormat %b

    section Phase 1 - Foundation
    Prediction Event schema           :p1a, 2026-03-01, 14d
    /predict with instrumentation     :p1b, after p1a, 21d
    /explain endpoint                 :p1c, after p1a, 14d
    Postgres schema + async writer    :p1d, after p1a, 14d
    /compare endpoint                 :p1e, after p1d, 14d
    /history endpoint                 :p1f, after p1d, 7d

    section Phase 2 - Monitoring
    Snowflake ingestion pipeline      :p2a, after p1e, 14d
    OOD detection + quality flags     :p2b, after p1b, 21d
    Basic drift monitoring queries    :p2c, after p2a, 14d
    Sandbox mode flag                 :p2d, after p1e, 14d

    section Phase 3 - Polish
    Support web UI                    :p3a, after p2d, 28d
    Interactive model explorer        :p3b, after p3a, 28d
    Alerting integration              :p3c, after p2c, 14d
```

---

## Notes

- All diagrams use Mermaid syntax and can be rendered in GitHub, VS Code, Quarto, or any Mermaid-compatible viewer.
- Diagrams reflect the **target state** architecture. Implementation will be iterative per the phase roadmap.
- See companion document `mlops_model_verification_discussion.md` for full design rationale, Q&A history, and decision log.
