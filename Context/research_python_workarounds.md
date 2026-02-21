# Python Workarounds for HDTweedie Tweedie Lasso: Agent 3 Research Report

*Authored by Agent 3 (theory/methods)*

---

## Background: What HDTweedie Actually Does

Before assessing alternatives, it is important to be precise about what HDTweedie delivers. The package, described in Qian et al. (2016) "Tweedie's Compound Poisson Model With Grouped Elastic Net" (*JCGS*, 25:606–625), implements the **IRLS-BMD algorithm**:

- **Outer loop (IRLS)**: At each iteration, approximate the Tweedie negative log-likelihood by a second-order Taylor expansion (a penalized weighted least squares problem), then update working responses and weights.
- **Inner loop (BMD — Blockwise Majorization Descent)**: Solve the penalized WLS subproblem via coordinate descent, exploiting the strong rule (Tibshirani et al.) to identify likely-zero coefficients and skip them.
- **Regularization path**: Fits a sequence of lambda values from `lambda_max` down to near-zero, producing a full coefficient path like glmnet.
- **Cross-validation**: `cv.HDtweedie` wraps the path with k-fold CV to select the optimal lambda.
- **Core**: The hot inner loop is Fortran, called from R via `.Fortran()`. The compiled Fortran handles coordinate updates with BLAS (OpenBLAS) acceleration.

The power parameter `p` is fixed by the user. For insurance compound Poisson-Gamma losses, `p` is typically in (1, 2).

**Critical gap in Python ecosystem**: scikit-learn's `TweedieRegressor` uses only L2 (ridge) regularization. There is **no L1 path** in sklearn for Tweedie. This is the fundamental gap all approaches below are trying to fill.

---

## Summary Comparison Table

| Rank | Approach | Fidelity | Complexity | Production-Ready | Docker | Trains? | Infers? |
|------|----------|----------|------------|-----------------|--------|---------|---------|
| 1 | **glum** (bonus find) | Very high | 1–3 days | High | Excellent | **Yes** | Yes |
| 2 | **Export coefficients** | Exact (infer) | 0.5 day | Highest | Excellent | No | Yes |
| 3 | rpy2 | Exact | 1–2 days | Moderate | +R runtime | Yes (via R) | Yes |
| 4 | XGBoost gblinear | Moderate | 1–4 weeks | High | Excellent | Partial | Yes |
| 5 | Custom NumPy CD | High | 3–6 weeks | Low initially | Yes | Yes | Yes |
| 6 | PyTorch/JAX | Moderate | 2–4 weeks | Moderate | Large image | Yes (approx) | Yes |
| 7 | CVXPY | High | 2–3 weeks | Low | Moderate | Yes (slow) | No |
| 8 | ONNX export | Exact (infer) | 0.5–1 day | High | Excellent | No | Yes |
| 9 | Cython/Fortran wrap | Exact | 4–8 weeks | Low | Complex | Yes | Yes |

---

## Recommended Strategy by Goal

**Goal: Python serving, R training (fastest, lowest risk)**
→ Use **Approach 8 / Export Coefficients**. Export the HDTweedie coefficient vector to JSON from R; score with numpy in Python. Keep the R training pipeline. Engineering cost: half a day.

**Goal: Full Python pipeline (training + inference)**
→ Use **glum**. It implements the same IRLS-CD algorithm with Tweedie + L1. Install via `pip install glum`. This is the only option that solves the training problem in pure Python without reimplementing the algorithm or introducing framework bloat.

**Goal: Temporary bridge while migrating**
→ Use **rpy2**. Two days to wrap HDTweedie calls in Python. Not a permanent solution, but allows the Python API layer to exist while training logic remains in R.

**Do not pursue**: Cython/Fortran wrapping (legal + engineering complexity), PyTorch/JAX (wrong tool for sparse linear models), CVXPY (proof-of-concept tool).

---

## BONUS: glum — The Direct Python Replacement

Before covering the 8 requested approaches, this finding deserves front placement.

**`glum`** (`pip install glum`) is a high-performance Python-first GLM library maintained by QuantCo (a commercial quantitative finance/insurance tech firm) that implements IRLS with coordinate descent — essentially the same algorithm as HDTweedie — with full support for:

- **Tweedie distribution** with arbitrary power `p` via `glum.TweedieDistribution(p)`
- **L1 lasso regularization** via `l1_ratio=1.0`
- **Elastic net** via `0 < l1_ratio < 1`
- **Regularization path** via `alpha_search=True`
- **Cross-validation** via `GeneralizedLinearRegressorCV`
- **Offsets and sample weights** — critical for insurance pricing
- **Scikit-learn-compatible API**

**Minimal HDTweedie equivalent in glum**:

```python
from glum import GeneralizedLinearRegressorCV, TweedieDistribution

cv_model = GeneralizedLinearRegressorCV(
    family=TweedieDistribution(1.5),   # p = 1.5
    l1_ratio=1.0,                       # pure lasso
    fit_intercept=True,
    cv=5,
    solver="irls-cd",                   # IRLS + coordinate descent (same as HDTweedie)
)
cv_model.fit(X_train, y_train, sample_weight=exposure_train)

print(cv_model.alpha_)   # selected lambda
print(cv_model.coef_)    # coefficient vector
print(cv_model.intercept_)
```

**Algorithmic correspondence to HDTweedie**:

| HDTweedie | glum |
|-----------|------|
| IRLS outer loop | IRLS outer loop |
| BMD (blockwise majorization descent) | Coordinate descent inner solver |
| Strong rule for active set | Active set tracking |
| Fortran + BLAS | Cython + BLAS (via numpy) |
| Lambda path, warm-start | `alpha_search=True`, warm-start |
| k-fold CV | `GeneralizedLinearRegressorCV` |

**Coefficients**: Numerically very close to HDTweedie for the same lambda, but not bit-for-bit identical. glum uses the true Tweedie Hessian (not Fisher information approximation), which benchmarks show converges in fewer IRLS iterations.

**Production notes**: Actively maintained, used in commercial insurance production, scikit-learn compatible, pure Python/Cython with pre-built wheels, tiny Docker footprint (dependencies: `numpy`, `scipy`, `tabmat`).

---

## Approach 1: Custom Coordinate Descent in NumPy/SciPy

### Algorithm

For a Tweedie GLM with log link, the IRLS-BMD algorithm works as follows. The working response and weight at current iterate `μ_i = exp(x_i'β)`:

```
Working response: z_i = x_i'β + (y_i - μ_i) / μ_i   [log link]
Working weight:   w_i = μ_i^(2-p)                     [var = μ^p, link deriv = μ]
```

Each IRLS outer iteration solves a penalized WLS subproblem (which is convex):

```
min_β  Σ_i w_i (z_i - x_i'β)² + λ ||β||₁
```

The coordinate descent update for predictor j:

```python
def soft_threshold(t, lam):
    return np.sign(t) * max(0, abs(t) - lam)

# Partial residual omitting predictor j:
r_j = z - X @ beta + X[:, j] * beta[j]

# Update:
numerator = np.sum(W * X[:, j] * r_j)
denominator = np.sum(W * X[:, j]**2)
beta[j] = soft_threshold(numerator, lam) / denominator
```

### Numerical Pitfalls

1. **Working weight collapse**: When `μ_i → 0`, `w_i = μ_i^(2-p)` underflows. Clamp `μ_i ≥ 1e-8`.
2. **Working response inflation**: When `μ_i ≈ 0` and `y_i > 0`, `z_i ≈ x_i'β + y_i/μ_i → ∞`. Clamp working responses and weights simultaneously.
3. **Non-convexity**: The Tweedie log-likelihood with log link is concave in β for `p ∈ (1,2)` (Hessian is PSD everywhere), but IRLS is still needed because the Hessian depends on the current iterate. Warm-starting down the lambda path is critical for numerical stability.
4. **Lambda max computation**: Estimate as `max_j |Σ_i w_i^(0) x_ij z_i^(0)| / n` at the null model (intercept only). Getting this wrong misaligns the regularization path.
5. **Exact zero coefficients**: Use a soft-threshold tolerance of ~1e-10 to handle floating-point noise near zero.

### Reference Implementations

- Qian et al. (2016) paper describes the full algorithm (Section 3)
- Wu & Lange (2008) "Coordinate Descent Algorithms for Lasso Penalized Regression" (*Annals of Applied Statistics*) — general reference
- `github.com/Quantco/glum/src/glum/` — best Python reference implementation to study
- `github.com/cran/HDtweedie/src/` — the actual Fortran core

### Assessment

| Dimension | Assessment |
|-----------|------------|
| Fidelity | High if implemented correctly |
| Complexity | **3–6 person-weeks** for numerically robust, tested implementation |
| Production readiness | Low initially — requires extensive numerical testing |
| Docker/MLOps | Excellent — pure numpy |
| Trains? | Yes |
| Infers? | Yes |

**Verdict**: Substantial undertaking. The algorithm is not exotic, but numerical edge cases (IRLS divergence, lambda path construction, cross-validation infrastructure) require careful engineering. The testing burden for an insurance-grade model is not trivial. `glum` has already solved all of these problems.

---

## Approach 2: PyTorch/JAX with Custom Tweedie Loss + L1

### Conceptual Setup

```python
import torch

def tweedie_nll(y_pred, y_true, p):
    # log link: y_pred = exp(X @ beta)
    a = y_true * torch.exp((1 - p) * torch.log(y_pred)) / (1 - p)
    b = torch.exp((2 - p) * torch.log(y_pred)) / (2 - p)
    return torch.mean(-a + b)

def tweedie_lasso_loss(beta, X, y, p, lam):
    mu = torch.exp(X @ beta)
    return tweedie_nll(mu, y, p) + lam * torch.sum(torch.abs(beta))
```

### Why Gradient-Based Methods Fail for Lasso

Coordinate descent is the dominant approach for lasso precisely because it handles the L1 subgradient **exactly** via soft-thresholding. Gradient-based optimizers approximate L1 via subgradients, producing coefficients that are small but **not exactly zero** — the hallmark of lasso is exact sparsity, which gradient methods cannot achieve without explicit post-processing.

- **Adam**: Not appropriate. Adaptive step sizes interfere with the proximity structure of L1. You get near-zero but not zero coefficients.
- **L-BFGS-B**: Can be adapted for lasso via proximal steps, but this is essentially reimplementing coordinate descent logic.
- **Proximal gradient (ISTA/FISTA)**: Legitimate approach in JAX via `jaxopt`. The update `β ← S(β - α∇ℒ(β), α·λ)` correctly achieves sparsity but converges slower than coordinate descent (O(1/k²) vs. effectively O(1/k) per parameter for CD).

### Performance Implications

For 20–50 predictors, PyTorch/JAX framework overhead is pure waste. These tools are designed for deep networks; the autograd machinery adds substantial overhead for 50-coefficient GLMs. The performance difference between a NumPy CD implementation and PyTorch on this problem size is negligible in computation time, but PyTorch brings a 500MB–2GB Docker image.

**A regularization path via gradient methods ≠ a coordinate descent path**: path shapes differ. For insurance pricing where the path is used to understand variable selection behavior, this matters for audit/review.

### Assessment

| Dimension | Assessment |
|-----------|------------|
| Fidelity | Moderate — comparable predictions, different path shape, no exact zeros |
| Complexity | **2–4 person-weeks** for proper proximal gradient with CV |
| Production readiness | Moderate — large Docker images, GPU/CPU env setup |
| Docker/MLOps | Large image footprint |
| Trains? | Yes (with sparsity caveats) |
| Infers? | Yes |

**Verdict**: Wrong tool for this job. Introduces deep learning infrastructure overhead for a problem that does not need it. If near-lasso behavior is acceptable, `glum` is strictly better.

---

## Approach 3: XGBoost/LightGBM in Linear Mode (`booster='gblinear'`)

### XGBoost gblinear

When `booster='gblinear'` is set, XGBoost fits a regularized linear model (not a tree ensemble):

```python
import xgboost as xgb

params = {
    "booster": "gblinear",
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.5,   # p
    "alpha": 0.1,    # L1 regularization
    "lambda": 0.0,   # L2 regularization (set to 0 for pure lasso)
    "updater": "coord_descent",
}
model = xgb.train(params, dtrain, num_boost_round=200)
```

Coefficients are extractable via `model.get_dump(dump_format='json')`. The prediction is `exp(X @ w + bias)` — fully interpretable as a GLM.

### Critical Caveats

1. **No regularization path**: XGBoost does not produce a path across lambda values. Each `alpha` requires a full separate fit from scratch (no warm-starting). You must implement the path loop manually — substantial work.
2. **No built-in cross-validation**: Manual implementation required.
3. **Boosting rounds ≠ convergence**: You must set `num_boost_round` high enough for coordinate descent to converge. There is no built-in convergence criterion.
4. **Feature scaling**: XGBoost does not standardize features internally. If predictors are on different scales, L1 disproportionately shrinks large-scale features. Pre-standardization is mandatory and the coefficients must be back-transformed.
5. **eta (step size)**: With `eta < 1`, convergence is slowed. Must set `eta = 1` for standard CD behavior.

### LightGBM

LightGBM does **not** have a `gblinear` equivalent. Its `lambda_l1` applies only to tree-based models (`gbdt`, `dart`). `linear_tree` adds linear models at leaves but is not globally linear. **LightGBM is not viable for this use case.**

### Assessment

| Dimension | Assessment |
|-----------|------------|
| Fidelity | Moderate — same Tweedie loss and L1, but no path or CV |
| Complexity | 1–2 weeks for a single model; **3–4 weeks** to add path + CV |
| Production readiness | High for the core XGBoost piece |
| Docker/MLOps | Excellent — small footprint |
| Trains? | Partially — single lambda only |
| Infers? | Yes |

**Verdict**: Legitimate for fitting a single Tweedie lasso model at one lambda value. Fails to replicate HDTweedie's core value proposition (regularization path + built-in CV). Engineering effort to reconstruct path + CV on top of gblinear exceeds just using `glum`.

---

## Approach 4: CVXPY or Convex Optimization

### Is Tweedie Deviance Convex?

The Tweedie log-likelihood with **log link** for `p ∈ (1,2)` is:

```
ℒ(β) = Σ_i [-y_i · exp((1-p)·x_i'β) / (1-p) + exp((2-p)·x_i'β) / (2-p)]
```

The Hessian with respect to `β_j`:

```
∂²ℒ/∂β_j² = Σ_i [(2-p)·x_ij²·μ_i^(2-p) - (1-p)·y_i·x_ij²·μ_i^(1-p)]
```

For `p ∈ (1,2)`: both `(2-p) > 0` and `-(1-p) > 0`, so both terms are non-negative. The Hessian is PSD everywhere — the log-likelihood **is concave in β** with log link for `p ∈ (1,2)`. The Tweedie lasso problem (negative log-likelihood + L1 penalty) **is a convex optimization problem**. CVXPY can in principle solve it.

### CVXPY Implementation

```python
import cvxpy as cp
import numpy as np

def tweedie_lasso_cvxpy(X, y, p, lam):
    n, d = X.shape
    beta = cp.Variable(d)
    eta = X @ beta  # linear predictor

    # Note: DCP compliance requires careful reformulation
    # cp.exp() is convex; multiply by negative y, then negate, is concave
    nll = cp.sum(
        -cp.multiply(y, cp.exp((1-p) * eta)) / (1-p)
        + cp.exp((2-p) * eta) / (2-p)
    ) / n

    penalty = lam * cp.norm1(beta)
    prob = cp.Problem(cp.Minimize(nll + penalty))
    prob.solve(solver=cp.ECOS)
    return beta.value
```

### DCP Compliance Issues

CVXPY requires problems to conform to Disciplined Convex Programming rules. `cp.multiply(y, cp.exp(...))` with `y > 0` involves multiplying a positive constant by a convex function (exp), producing a convex term — when negated, this is concave, which DCP allows under minimization only if the concave term appears with a positive coefficient. The formulation above should be DCP-compliant, but requires careful verification that CVXPY's rule checker accepts it.

### Performance Issues

CVXPY calls general-purpose interior-point solvers (ECOS, SCS, Mosek). For a regularization path (100 lambda values), this means 100 cold-start interior-point problems. Interior-point methods do not warm-start as efficiently as coordinate descent — expect 10–100× slower than glum for path computation.

### Assessment

| Dimension | Assessment |
|-----------|------------|
| Fidelity | High (same objective, convex) if DCP issues resolved |
| Complexity | **2–3 person-weeks** for DCP formulation + path + CV |
| Production readiness | Low — CVXPY is a modeling prototyping tool |
| Docker/MLOps | Moderate — CVXPY/ECOS has C dependencies |
| Trains? | Yes (slowly) |
| Infers? | No — CVXPY is not a scoring API |

**Verdict**: Proof-of-concept tool, not a production fitting engine. Produces the correct optimal solution but at 10–100× the cost of a specialized solver. For auditing that the optimization is correct, CVXPY is excellent. For production training pipelines, use `glum`.

---

## Approach 5: Cython/pybind11 Wrapping of HDTweedie's Fortran Core

### What the Fortran Core Does

The CRAN mirror (`github.com/cran/HDtweedie`) shows the R package calls `.Fortran("tweediegrpnet", ...)`. The Fortran subroutine implements the inner coordinate descent loop (approximately 200 lines). The R wrapper (`R/HDtweedie.R`) handles data preparation, lambda path construction, warm-starting, and cross-validation (approximately 500+ lines).

### What Wrapping Involves

1. Extract Fortran source from the CRAN package tarball
2. Write a C interface matching R's `.Fortran()` calling conventions
3. Write Python bindings via Cython or `cffi`
4. **Reimplement all R wrapper logic in Python** — the IRLS outer loop, lambda path construction, cross-validation infrastructure

### Why This Is the Worst Option

- **GPL-2 license**: HDTweedie is GPL-2. Using the Fortran source in a Python extension propagates GPL-2 to your wrapper. For proprietary insurance code, this is a legal concern requiring formal review.
- **80% of the value is in the R wrapper**: You must rewrite it anyway, so you pay the algorithmic implementation cost plus the Fortran wrapping cost.
- **Build complexity**: Fortran compiler + OpenBLAS linkage required in Docker. Fragile across architectures (x86 vs. ARM).
- **Maintenance**: Any HDTweedie update requires re-wrapping.
- **At 20–50 predictors, the Fortran core is not the bottleneck**: The IRLS outer loop and data conversion are.

### Assessment

| Dimension | Assessment |
|-----------|------------|
| Fidelity | Exact (uses same numerical core) |
| Complexity | **4–8 person-weeks** |
| Production readiness | Low — fragile build, GPL licensing |
| Docker/MLOps | Complex — Fortran compiler + BLAS in container |
| Trains? | Yes |
| Infers? | Yes |

**Verdict**: Worst option. All the complexity of Fortran wrapping plus reimplementing R business logic, with GPL entanglement. No scenario justifies this for a 20–50 predictor problem.

---

## Approach 6: rpy2 — Calling R from Python

### What rpy2 Does

rpy2 embeds an R interpreter within a Python process. You can call HDTweedie directly:

```python
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import numpy as np

ro.r('library(HDtweedie)')
hdtweedie = importr('HDtweedie')
ro.numpy2ri.activate()

result = hdtweedie.cv_HDtweedie(X, y, p=1.5)
beta_selected = np.array(result.rx2('glmnet.fit').rx2('beta'))
```

### Installation and Docker

```dockerfile
FROM python:3.12
RUN apt-get install -y r-base libr-dev   # ~500MB addition to image
RUN pip install rpy2
RUN Rscript -e "install.packages('HDtweedie')"
```

R package installation at build time adds significant image size and build time. Pinning both R version and rpy2 version is mandatory — rpy2 has historically had version compatibility issues.

### Performance Overhead

- First call incurs ~0.5–1 second R initialization overhead
- Data conversion between R (SEXP objects) and Python (NumPy arrays) is fast for this data size
- Actual fitting time dominates; rpy2 overhead is negligible in practice

### Production Reliability Concerns

- R garbage collection can interfere with Python's memory manager
- Segfaults reported when R and Python simultaneously manage memory regions
- R is fork-unfriendly: Python's `multiprocessing` can cause instability
- R signal handlers conflict with Python's

### Does This Defeat the Purpose?

If the goal is **eliminating R from the stack**, rpy2 does not achieve it — it hides R behind a Python function call. If the goal is **a Python training pipeline that calls out to R for this one step**, rpy2 is pragmatic and two days of engineering cost.

### Assessment

| Dimension | Assessment |
|-----------|------------|
| Fidelity | Exact |
| Complexity | **1–2 person-days** to get working |
| Production readiness | Moderate — workable but fragile |
| Docker/MLOps | Adds R runtime (~500MB), version pinning required |
| Trains? | Yes (via R) |
| Infers? | Yes (via R) |

**Verdict**: Fastest path to calling HDTweedie from Python code. Does not eliminate R dependency. Best as a temporary bridge or for Python orchestration pipelines where most logic is Python but one model training step uses R. Not appropriate if the goal is full R elimination.

---

## Approach 7: ONNX or Model Serialization

### What ONNX Offers

ONNX supports linear model operators (`LinearRegressor`) and basic math operations. A Tweedie GLM is `exp(X @ beta + intercept)`, expressible as `MatMul → Add → Exp` in ONNX. sklearn-onnx can convert `TweedieRegressor` objects to ONNX, but `TweedieRegressor` only supports L2 regularization — so you cannot train with L1 in sklearn and export to ONNX for HDTweedie-equivalent inference.

**PMML**: R's `pmml` package can export fitted GLMs. Python can consume PMML via `sklearn-pmml-model`. This is a viable serialization format for an HDTweedie-fitted model.

### The Core Issue

ONNX and PMML solve the **inference problem only**. They have nothing to do with training. The workflow is:

1. Fit in R using HDTweedie
2. Export fitted coefficients
3. Decode prediction formula in ONNX/PMML/numpy

For a 50-coefficient linear model, ONNX is engineering overkill that adds a runtime dependency and schema complexity. The coefficient-export approach (Approach 8) achieves identical results with zero additional dependencies.

### Assessment

| Dimension | Assessment |
|-----------|------------|
| Fidelity | Exact (inference only) |
| Complexity | **0.5–1 person-day** |
| Production readiness | High |
| Docker/MLOps | Excellent |
| Trains? | No |
| Infers? | Yes |

**Verdict**: Technically correct but over-engineered. Approach 8 (coefficient export) is strictly simpler.

---

## Approach 8: The "Just Export Coefficients" Approach

### How Simple Is This?

For a Tweedie GLM with log link, the complete inference specification is:

```python
import numpy as np
import json

with open("model_coefficients.json") as f:
    model = json.load(f)

beta = np.array(model["beta"])
intercept = float(model["intercept"])
feature_names = model["feature_names"]  # for audit trail

def predict(X):
    """Predict expected loss using Tweedie GLM (log link)."""
    return np.exp(X @ beta + intercept)

def predict_with_exposure(X, exposure):
    """Predict expected loss with exposure offset."""
    return exposure * np.exp(X @ beta + intercept)
```

This is the **complete** scoring implementation. The log-link Tweedie GLM prediction is `exp(eta)` where `eta = X @ beta`. No hidden state, no distribution-specific sampling, no complexity.

### Coefficient Export from R

```r
library(HDtweedie)

cv_fit <- cv.HDtweedie(X, y, p = 1.5)
coefs <- coef(cv_fit, s = "lambda.min")
beta_vector <- as.numeric(coefs[-1])   # exclude intercept
intercept_val <- as.numeric(coefs[1])

export <- list(
  beta = beta_vector,
  intercept = intercept_val,
  features = colnames(X),
  lambda = cv_fit$lambda.min,
  tweedie_p = 1.5,
  timestamp = as.character(Sys.time())
)

jsonlite::write_json(export, "model_coefficients.json", auto_unbox = TRUE)
```

### What You Get

- **Zero production dependencies** beyond numpy
- **Auditable**: every coefficient is a named number in a JSON file — regulators can review it
- **Maximally portable**: runs on any Python 3.x environment
- **Interpretable**: actuaries can read the coefficient file and understand the model
- **Reproducible**: the JSON file IS the model; git-versioning it gives full reproducibility
- **Operational workflow**: R training job → export JSON → deploy Python scoring service

### What You Give Up

- Training pipeline stays in R (not necessarily a loss — it means the training environment stays stable while the serving environment is pure Python)
- Re-export required after retraining (requires an operational workflow)

### Assessment

| Dimension | Assessment |
|-----------|------------|
| Fidelity | Exact (for inference); training stays in R |
| Complexity | **0.5 person-day** |
| Production readiness | Highest of all approaches |
| Docker/MLOps | Zero additional dependencies |
| Trains? | No — training stays in R |
| Infers? | Yes, perfectly |

**Verdict**: The right answer for production insurance model serving if the constraint is "Python serving" rather than "Python training." The training pipeline in R with HDTweedie is the hard part — and R is already doing it correctly. Export coefficients; score in numpy. Done.

---

## Also Worth Knowing: statsmodels GLM.fit_regularized

`statsmodels.genmod.GLM.fit_regularized()` supports L1 regularization for GLMs, including Tweedie family (via `sm.families.Tweedie(var_power=p, link=sm.families.links.log())`). However:

- Does not produce a regularization path
- Does not have built-in cross-validation
- Known numerical instability issues with Tweedie (GitHub issue #7476)
- Uses elastic net objective but the coordinate descent implementation is less robust than glum

Not recommended as a primary approach, but useful for quick verification of coefficient magnitudes at a single lambda.

---

## Appendix: Convexity Proof for Tweedie with Log Link

For completeness, the Tweedie negative log-likelihood with log link is convex in β for `p ∈ (1,2)`:

The Hessian diagonal element for predictor j:

```
H_jj = Σ_i x_ij² · [(2-p)·μ_i^(2-p) - (1-p)·y_i·μ_i^(1-p)]
```

For `p ∈ (1,2)`: `(2-p) > 0` always, and `-(1-p) = (p-1) > 0` always. Since `μ_i > 0` and `y_i ≥ 0`, both terms are non-negative, so `H_jj ≥ 0`. The Hessian is PSD everywhere, confirming global convexity. This means coordinate descent, IRLS, proximal gradient, and interior-point methods all converge to the same global optimum.

---

*Sources*: Qian et al. (2016) JCGS 25:606–625; Wu & Lange (2008) Annals Applied Statistics; glum documentation (Quantco, readthedocs.io); XGBoost documentation; rpy2 documentation; CVXPY documentation; HDtweedie CRAN documentation.
