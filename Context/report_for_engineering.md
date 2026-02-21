# Tweedie GLM Deployment: Engineering Architecture Review

**Prepared for:** Platform Engineering / MLOps Team
**Date:** 2026-02-19
**Subject:** Production deployment options for the Tweedie Lasso GLM pricing model

---

## Executive Summary

We investigated whether our insurance pricing model (a Tweedie lasso GLM currently running in R) should be rewritten in Python for production deployment. After exhaustive research into the Python ecosystem, we found that **no Python package is a complete 1:1 replacement** for the R package (HDTweedie) that fits this model. The closest alternative (`glum` by QuantCo) covers ~90% of functionality and is a credible option if a Python training pipeline is required. However, the current R + Plumber + Docker deployment is a production-grade architecture that meets every MLOps requirement — and rewriting the model creates engineering work with no measurable platform benefit.

This document presents the options objectively and addresses engineering concerns directly.

---

## 1. What the Model Actually Does at Scoring Time

Before discussing deployment architecture, it's important to understand what the model does in production. At inference time, the Tweedie GLM with log link computes:

```
predicted_loss = exp(intercept + beta_1 * x_1 + beta_2 * x_2 + ... + beta_p * x_p)
```

Or with an exposure offset (standard in insurance pricing):

```
predicted_loss = exposure * exp(intercept + beta_1 * x_1 + ... + beta_p * x_p)
```

That's it. **A dot product, an addition, and `exp()`.** With ~20-50 features, this computation takes approximately **3 microseconds** in R. The Tweedie distribution, the lasso penalty, the cross-validation, the regularization path — all of that complexity lives in the *training* phase, not the *scoring* phase. In production, the model is just a vector of coefficients and one line of arithmetic.

This matters because most engineering concerns about R relate to training complexity or language unfamiliarity — neither of which affects the scoring API.

---

## 2. The Four Deployment Options

We evaluated four approaches, ordered by engineering effort:

### Option A: Keep R + Plumber + Docker (Current State)

**Effort: 0 days.** The model is already deployed and working.

**How it works:**
- R Plumber API serves predictions over HTTP (JSON in, JSON out)
- Containerized with Docker using `rocker/r-ver` base images
- Packages pinned via `renv.lock` for full reproducibility

**Performance:**
- Plumber: ~1,075 req/s at 18.6 ms average latency (single process)
- GLM scoring computation: ~3 microseconds (0.02% of request latency)
- The remaining 18.597 ms is HTTP overhead — identical to what Flask/FastAPI adds
- With RestRserve (drop-in replacement for plumber): 20,000+ req/s on 4-core hardware

**Scaling:**
- Kubernetes HPA scales pods horizontally — 10 pods = 10,000 req/s with plumber
- This is the same scaling model as Flask with Gunicorn workers or FastAPI with Uvicorn
- For our pricing workload (thousands of predictions/day), a single pod is sufficient

### Option B: Export Coefficients to Python (Inference Only)

**Effort: 0.5 days.** Training stays in R. Scoring moves to Python.

**How it works:**
1. After training in R, export the coefficient vector to JSON:

```r
# R: Export model coefficients
cv_fit <- cv.HDtweedie(X, y, p = 1.5)
coefs <- coef(cv_fit, s = "lambda.min")

export <- list(
  intercept = as.numeric(coefs[1]),
  beta = as.numeric(coefs[-1]),
  features = colnames(X),
  tweedie_p = 1.5,
  model_version = "v2.1.0",
  trained_at = as.character(Sys.time())
)
jsonlite::write_json(export, "model_coefficients.json", auto_unbox = TRUE)
```

2. In Python, load and score:

```python
# Python: Score new observations
import numpy as np
import json

with open("model_coefficients.json") as f:
    model = json.load(f)

beta = np.array(model["beta"])
intercept = model["intercept"]

def predict(X: np.ndarray) -> np.ndarray:
    """Predict expected loss cost. X is (n_samples, n_features)."""
    return np.exp(X @ beta + intercept)

def predict_with_exposure(X: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """Predict expected loss with exposure offset."""
    return exposure * np.exp(X @ beta + intercept)
```

**Why this gives exact numerical equivalence:** The Python scoring code uses the *identical* coefficients produced by HDTweedie. Matrix multiplication and `exp()` are deterministic — results match R to within floating-point epsilon (~10^-15). No statistical estimation occurs in Python; it's purely arithmetic.

**What it gives engineering:**
- Pure Python scoring service — no R dependency in the scoring container
- Zero additional dependencies beyond NumPy
- Model is a versioned JSON file — git-trackable, auditable, trivially portable
- FastAPI/Flask serving with full Python tooling ecosystem

**What it requires:**
- Training remains in R (actuaries retrain quarterly/annually, export new JSON)
- An operational workflow for the R-training -> JSON-export -> Python-deploy pipeline

**Important caveat:** This approach covers point prediction only (expected loss cost). If the deployment ever needs prediction intervals, distributional forecasts, or claim simulation, additional parameters (power parameter `p`, dispersion `phi`) must be exported and a Tweedie distribution implementation is needed in the Python scoring layer. This would add moderate complexity. For pure pricing (which is our current use case), point prediction is sufficient.

### Option C: Migrate Training to glum (Full Python)

**Effort: 1-3 days for API migration + validation testing.**

`glum` (by QuantCo, `pip install glum`) is the closest Python alternative to HDTweedie. It implements the same algorithm family (IRLS with coordinate descent) for Tweedie distributions with L1/lasso penalization:

```python
from glum import GeneralizedLinearRegressorCV, TweedieDistribution

model = GeneralizedLinearRegressorCV(
    family=TweedieDistribution(power=1.5),
    l1_ratio=1.0,           # pure lasso
    solver="irls-cd",       # IRLS + coordinate descent
    cv=5,
    fit_intercept=True,
)
model.fit(X_train, y_train, sample_weight=exposure)
```

**Gaps vs. HDTweedie:**
- No native full coefficient path matrix in one call (needs a loop wrapper — minor)
- Coefficients will be *close* but not *identical* to HDTweedie — `glum` uses the true Tweedie Hessian while HDTweedie uses Fisher information, causing slightly different IRLS convergence paths
- No grouped lasso (not needed for our model — we use standard lasso)

**The validation question:** Because coefficients differ (same algorithm family, different implementation details), migrating to `glum` triggers a model re-validation. The actuarial team must run the R and Python models side-by-side on the full portfolio and document that differences are within acceptable tolerance. This is non-trivial — see the actuarial management report for regulatory details.

### Option D: Full Python Port of HDTweedie

**Effort: 15-25 person-weeks.**

Not recommended. The HDTweedie R package is 508 lines of Fortran 90 + ~450 lines of R. A full port to Python/Cython requires:
- Deep understanding of the IRLS-BMD algorithm and Tweedie deviance numerics
- Extensive numerical validation against the R version
- Long-term maintenance ownership by our team
- GPL-2 licensing constraints (HDTweedie is GPL-2; a derivative work inherits this)

When `glum` already exists and covers our use case, a custom port is not justified.

---

## 3. Addressing Engineering Concerns Directly

### "We don't know R and can't maintain it"

The engineering team does not maintain R code. The responsibilities are cleanly separated:

| Responsibility | Owner | Language Knowledge Required |
|---|---|---|
| Model code (R) | Actuarial team | R |
| Dockerfile | Actuarial team (30 lines, mostly `renv::restore()`) | Dockerfile |
| CI/CD pipeline | Engineering | None (builds image, pushes, deploys) |
| Kubernetes manifests | Engineering | YAML |
| Monitoring/alerting | Engineering | PromQL, Grafana |
| Incident response | Engineering | HTTP, logs, pod restart |

The engineering team never reads or modifies R code. They operate containers. Container operations are identical regardless of the language inside.

### "R is slow"

For the actual workload — scoring a 20-50 feature GLM — R computes `exp(X %*% beta)` in 3 microseconds. The difference between plumber (18.6 ms) and FastAPI (11 ms) is 7.6 ms of HTTP framework overhead. This is:
- Invisible to any human user
- Irrelevant to any async batch pipeline
- Smaller than network jitter between services

If throughput becomes a constraint, the solution is horizontal scaling (more pods), not a language rewrite. Python Flask apps use the exact same scaling model.

### "R isn't production-grade"

Companies running R in production:

| Organization | Use Case |
|---|---|
| **Swiss Re** | Production R for risk management (presented at R/Insurance 2024) |
| **T-Mobile** | Customer-facing production API (plumber + Docker) |
| **Roche** | GxP-compliant Posit Stack on Kubernetes |
| **10+ pharma majors** | FDA regulatory submissions in R (Pilots 1-4, 2021-2024) |
| **NHS** | Public-facing Shiny dashboards (since April 2023) |
| **ANZ Bank** | Credit risk modeling in production |

Microsoft, Google, and Oracle are all paying members of the R Consortium. Posit provides enterprise tooling (Connect, Workbench, Package Manager) with SLAs and Kubernetes integration.

### "Our MLOps tools won't work with R"

Every MLOps tool operates at the container level. Here's what "R in Docker" looks like to each tool:

**Kubernetes:** An OCI image with an HTTP endpoint. The Deployment manifest, HPA config, and health probes are identical for R and Python.

**Prometheus:** R plumber exposes `/metrics` via the `openmetrics` package (3 lines of setup code). Prometheus scrapes it identically to a Python endpoint.

```r
library(openmetrics)
srv <- register_plumber_metrics(plumber::plumb("api.R"))
```

**Structured logging:** R writes newline-delimited JSON via the `logger` package. ELK/Loki/CloudWatch ingests it identically to Python JSON logs.

```r
library(logger)
log_formatter(formatter_json)
log_info(method = req$REQUEST_METHOD, path = req$PATH_INFO)
```

**MLflow:** R models are tracked and registered in MLflow identically to Python models. Container image tag = model version.

**CI/CD (GitHub Actions, ArgoCD, Jenkins):** Builds a Docker image, pushes to registry, updates deployment. The pipeline does not parse Dockerfiles or detect languages.

**Canary deployments (Flagger/Istio):** Traffic splitting between container versions based on Prometheus metrics. Language-agnostic by design.

### "What if we need to call the model from Python services?"

The model is already an HTTP API. Any Python service calls it with:

```python
import requests
response = requests.post(
    "http://scoring-api:8080/predict",
    json={"features": feature_dict}
)
prediction = response.json()["predicted_loss"]
```

The caller does not know what processes the request. This is standard microservice architecture — the API contract (JSON schema) is the interface, not the implementation language.

### "We can't hire R developers"

You're not hiring R developers for DevOps. The org chart looks like this:

- **Actuaries / actuarial data scientists** (already know R) -> maintain the model
- **Platform engineers** (Python/Go/K8s) -> maintain the deployment infrastructure

These are separate hiring pools. The platform engineers never touch R. The actuaries never touch Kubernetes. Requiring actuaries to learn Python GLM libraries creates a productivity drag and increases error risk.

---

## 4. The Docker Architecture (For Reference)

### Production Dockerfile

```dockerfile
# Stage 1: Build (cached in CI)
FROM rocker/r-ver:4.5.0 AS builder
RUN apt-get update && apt-get install -y \
    libssl-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
COPY renv.lock renv.lock
RUN Rscript -e 'install.packages("renv"); renv::restore()'

# Stage 2: Production (lean)
FROM rocker/r-ver:4.5.0
COPY --from=builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library
COPY model/ /app/model/
COPY R/ /app/R/
COPY plumber.R /app/

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD curl --silent --fail http://localhost:8080/_ping || exit 1

EXPOSE 8080
CMD ["Rscript", "-e", \
  "pr <- plumber::plumb('/app/plumber.R'); pr$run(port=8080, host='0.0.0.0')"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tweedie-scoring
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: scoring-api
        image: registry.example.com/tweedie-glm:v2.1.0
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet: { path: /_ping, port: 8080 }
        readinessProbe:
          httpGet: { path: /_ping, port: 8080 }
        resources:
          requests: { memory: "512Mi", cpu: "250m" }
          limits: { memory: "1Gi", cpu: "500m" }
```

This manifest is **identical** regardless of whether the container runs R or Python. The orchestration layer does not know or care.

---

## 5. Recommendation

### If the goal is "Python in the scoring path":

**Use Option B (Export Coefficients).** Half a day of work. The scoring service is pure Python + NumPy. The R dependency is isolated to the training pipeline, which the actuarial team owns. No model re-validation required because the coefficients are identical.

### If the goal is "no R anywhere":

**Use Option C (glum) for training + Option B for serving.** 1-3 days of migration work, plus actuarial validation time. Be aware that coefficients will differ slightly from the R model, triggering a model change process under regulatory guidance. The actuarial team will need to assess and sign off.

### If the goal is "best engineering decision":

**Use Option A (Keep R + Plumber + Docker).** The model works. The container meets every MLOps requirement. The engineering investment should go into operationalization (Prometheus metrics, structured logging, Kubernetes health probes, renv lockfile hygiene) — not replatforming.

---

## 6. Comparison Table

| Criterion | Option A: R + Docker | Option B: Export Coefs | Option C: glum | Option D: Full Port |
|---|---|---|---|---|
| Engineering effort | 0 | 0.5 days | 1-3 days + validation | 15-25 weeks |
| Scoring in Python | No | Yes | Yes | Yes |
| Training in Python | No | No | Yes | Yes |
| Numerical equivalence | Exact | Exact (10^-15) | Close (~10^-4) | Untested |
| Re-validation needed | No | No | Yes | Yes |
| R dependency | Scoring + training | Training only | None | None |
| Production risk | None (status quo) | Very low | Low-moderate | High |
| MLOps compatibility | Full | Full | Full | Full |
| Long-term maintenance | Actuarial team | Actuarial + small Python wrapper | Engineering + actuarial | Engineering owns |

---

## Appendix: Python Ecosystem Assessment Summary

We exhaustively evaluated every Python package that could potentially replace HDTweedie:

| Package | Tweedie Support | L1 Lasso | Solution Path | CV | Verdict |
|---|---|---|---|---|---|
| **glum** (QuantCo) | Yes | Yes | Partial (loop) | Yes | Best alternative (~90%) |
| **h2o** | Yes | Yes | Yes | Yes | Requires JVM + H2O cluster |
| scikit-learn TweedieRegressor | Yes | **L2 only** | No | No | Cannot do lasso |
| statsmodels GLM | Approximate | Unstable | No | No | Known Tweedie+L1 bug (#7476) |
| glmnet-python (all variants) | **No** | Yes | Yes | Yes | No Tweedie family |
| pyglmnet | **No** | Yes | No | No | No Tweedie family |
| XGBoost gblinear | Yes | Yes | No | No | No path or CV built-in |
| PySpark GLM | Yes | Yes | No | No | Requires Spark cluster |

The gap is clear: Python's general-purpose ML ecosystem does not offer penalized (lasso) regression with Tweedie deviance as a first-class feature. `glum` is the exception, purpose-built by a company (QuantCo) that works in insurance pricing and faced exactly the same gap.

---

*This report synthesizes findings from five parallel research workstreams investigating HDTweedie internals, Python alternatives, workaround approaches, transcoding feasibility, and R production deployment patterns. Full source reports available in the `Context/` directory.*
