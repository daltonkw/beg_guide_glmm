# HDtweedie R Package: A Deep Technical Report

**Agent 1 — tweedie-research team**
**Date:** 2026-02-19
**Purpose:** Comprehensive technical analysis for evaluating Python replication of HDtweedie

---

## Table of Contents

1. [Package Identity and Provenance](#1-package-identity-and-provenance)
2. [The Foundational Academic Papers](#2-the-foundational-academic-papers)
3. [Mathematical Foundation: The Tweedie Distribution](#3-mathematical-foundation-the-tweedie-distribution)
4. [The Compound Poisson-Gamma Nature](#4-the-compound-poisson-gamma-nature)
5. [The Penalized Estimation Problem](#5-the-penalized-estimation-problem)
6. [The IRLS-BMD Algorithm](#6-the-irls-bmd-algorithm)
7. [How HDtweedie Differs from glmnet](#7-how-hdtweedie-differs-from-glmnet)
8. [Package API and Features](#8-package-api-and-features)
9. [Convergence Properties](#9-convergence-properties)
10. [Why Actuaries Rely on It](#10-why-actuaries-rely-on-it)
11. [Relationship to Sibling Packages](#11-relationship-to-sibling-packages)
12. [Python Replication Assessment](#12-python-replication-assessment)
13. [References](#13-references)

---

## 1. Package Identity and Provenance

| Field | Value |
|-------|-------|
| **Full name** | HDtweedie: The Lasso for Tweedie's Compound Poisson Model Using an IRLS-BMD Algorithm |
| **CRAN URL** | https://cran.r-project.org/web/packages/HDtweedie/ |
| **Version** | 1.2 (released May 10, 2022) |
| **License** | GPL-2 |
| **NeedsCompilation** | Yes — core is Fortran 90 |
| **Authors** | Wei Qian (Rochester Institute of Technology), Yi Yang (McGill University), Hui Zou (University of Minnesota) |
| **Monthly downloads** | ~362 (Jan 2026) |
| **Reverse dependencies** | `personalized2part` (Depends), `adaptMT` (Suggests) |

The package name "HD" stands for **High-Dimensional** — the algorithm was designed for settings where the predictor count `p` may be large relative to the sample size `n`. The Fortran routine at the core is named `tweediegrppath()`.

---

## 2. The Foundational Academic Papers

### Primary Paper

**Qian, W., Yang, Y., Yang, Y., and Zou, H. (2016).** "Tweedie's Compound Poisson Model With Grouped Elastic Net." *Journal of Computational and Graphical Statistics*, **25**(2), 606–625.

- DOI: https://doi.org/10.1080/10618600.2015.1005213
- Preprint (free): https://repository.rit.edu/cgi/viewcontent.cgi?article=2822&context=article
- Scopus citations: 31

**Keywords:** Coordinate descent, IRLS-BMD, Insurance score, Lasso, Variable selection

This is the definitive paper. It introduces the IRLS-BMD algorithm for penalized Tweedie regression, proves convergence, demonstrates variable selection on insurance data, and provides the theoretical justification for the BMD inner loop.

### Secondary Paper (Inner Algorithm Foundation)

**Yang, Y. and Zou, H. (2015).** "A fast unified algorithm for solving group-lasso penalized learning problems." *Statistics and Computing*, **25**, 1129–1141.

- DOI: https://doi.org/10.1007/s11222-014-9498-5

This earlier paper introduces the **GMD/BMD framework** — the general blockwise majorization descent method that HDtweedie extends to Tweedie GLMs. The `gglasso` R package implements BMD for standard GLMs (Gaussian, logistic, etc.); HDtweedie wraps it in an IRLS outer loop to handle the non-standard Tweedie likelihood.

### Tertiary Paper (Follow-up / GBM Extension)

**Yang, Y., Qian, W., and Zou, H. (2018).** "Insurance premium prediction via gradient tree-boosted Tweedie compound Poisson models." *Journal of Business & Economic Statistics*, **36**(3), 456–470.

This paper is by the same team and extends the Tweedie framework to gradient boosting (XGBoost), rather than linear penalized regression. It is the intellectual successor that motivated the `xgboost` Tweedie loss. Relevant as a comparison point for non-linear modeling.

### Foundational Reference: glmnet

**Friedman, J., Hastie, T., and Tibshirani, R. (2010).** "Regularization paths for generalized linear models via coordinate descent." *Journal of Statistical Software*, **33**(1), 1–22.

HDtweedie explicitly states it modifies the API and code structure from glmnet. Understanding glmnet's design is prerequisite to understanding HDtweedie's departures from it.

---

## 3. Mathematical Foundation: The Tweedie Distribution

### Exponential Dispersion Model (EDM) Framework

The Tweedie distributions form a sub-family of **exponential dispersion models** characterized by the **power variance function**:

```
Var(Y) = φ · μ^p
```

where:
- `μ = E[Y]` is the mean
- `φ > 0` is the **dispersion parameter** (scale, not location)
- `p` is the **power parameter** (Tweedie index)

The variance is a power of the mean — a defining property that makes the Tweedie family extremely flexible.

### The Power Parameter Zoo

| p value | Distribution | Key Property |
|---------|-------------|--------------|
| 0 | Normal (Gaussian) | Constant variance |
| 1 | Poisson | Variance = mean |
| **(1, 2)** | **Compound Poisson-Gamma** | **Point mass at 0 + continuous positive** |
| 2 | Gamma | CV constant (variance/mean² constant) |
| 3 | Inverse Gaussian | Heavy right tail |
| p > 3 | Stable distributions | Theoretical interest only |

The case `1 < p < 2` is the **central case for insurance** — it is the only distribution in the Tweedie family with a point mass at zero combined with a continuous positive density, exactly matching aggregate claim data.

### The Tweedie Log-Likelihood

For a single observation `y` with mean `μ` and dispersion `φ`, the Tweedie log-likelihood is:

```
ℓ(μ; y, φ, p) = (1/φ) · [y · μ^(1-p) / (1-p)  -  μ^(2-p) / (2-p)]  +  c(y, φ, p)
```

where `c(y, φ, p)` is a normalizing constant that is **infinite series** for `1 < p < 2`:

```
c(y, φ, p) = -log(y) + sum_{j=1}^{∞} log(y^j / (φ^(j·(2-p)/(p-1)) · j! · Γ(j·α)))
```

where `α = (2-p)/(p-1)` is the Gamma shape parameter. This series converges but has no closed form — evaluating the full likelihood requires either:
1. **Saddle-point approximation** (Nelder and Pregibon 1987; Dunn and Smyth 2005)
2. **Fourier inversion** (Dunn and Smyth 2008) — more accurate but slower

**This is the critical difficulty:** there is no closed-form density for `1 < p < 2`. The Tweedie distribution cannot be expressed as a simple exponential family density in computable terms.

### The Tweedie Deviance (Unit Deviance)

Despite the intractable density, the **deviance** is computable in closed form. The unit deviance function (the saturated-model comparison) for `p ≠ 1, 2` is:

```
d(y, μ) = 2 · [ y^(2-p) / ((1-p)(2-p))  -  y · μ^(1-p) / (1-p)  +  μ^(2-p) / (2-p) ]
```

The total deviance is:

```
D = Σᵢ wᵢ · d(yᵢ, μᵢ)
```

**This is what HDtweedie optimizes** — the penalized deviance, not the penalized log-likelihood. Since `c(y, φ, p)` is constant with respect to `β` (the regression coefficients), minimizing deviance is equivalent to maximizing the log-likelihood when `φ` is treated as fixed. HDtweedie does **not** estimate `φ` or profile over it — these are treated as given.

---

## 4. The Compound Poisson-Gamma Nature

### Exact Stochastic Representation

For `1 < p < 2`, the Tweedie distribution `Tw(μ, φ, p)` has the exact stochastic representation:

```
Y = X₁ + X₂ + ... + X_N
```

where:
- `N ~ Poisson(λ)` — number of claims (frequency)
- `X_i ~ Gamma(α, β)` i.i.d. — individual claim amounts (severity)
- `N` and the `{X_i}` are independent

The parameters link to the Tweedie parameters `(μ, φ, p)` by:

```
λ = μ^(2-p) / ((2-p) · φ)          [Poisson intensity: expected claim count]
α = (2-p) / (p-1)                    [Gamma shape: controls severity skewness]
β = φ · (p-1) · μ^(p-1)             [Gamma rate parameter]
```

These imply:
- `E[Y] = E[N] · E[X] = λ · α/β = μ` ✓
- `Var[Y] = E[N] · Var[X] + Var[N] · (E[X])² = φ · μ^p` ✓
- `P(Y = 0) = P(N = 0) = exp(-λ) = exp(-μ^(2-p) / ((2-p)φ))`

### The Probability of Zero

```
P(Y = 0) = exp(-μ^(2-p) / ((2-p) · φ))
```

In an insurance context:
- When `μ` (expected loss) is large, `P(Y=0)` is small — high-risk policyholders are unlikely to have zero claims
- When `μ` is small, `P(Y=0)` approaches 1 — low-risk policyholders almost always have no claims
- This natural coupling of frequency (N) and severity (X) through a single mean parameter `μ` is the key actuarial advantage

### Why This Matters for Insurance Pricing

In non-life insurance (property/casualty), aggregate annual claims per policy follow exactly this structure:

1. **Zero-inflation is structural, not statistical artifact.** Many policyholders file no claims in a given year. The probability of zero is determined by the Poisson frequency model, not an ad hoc zero-inflation mechanism.

2. **Frequency and severity are naturally coupled.** The single parameter `μ = exp(x'β)` governs both the expected number of claims and the expected size of each claim through the compound Poisson-Gamma decomposition. You don't need separate models.

3. **GLM is sufficient.** Because Y is a Tweedie EDM, standard GLM machinery (log link, IRLS, deviance-based inference) applies directly. No two-stage modeling is required.

4. **The log link ensures positivity.** With `μ = exp(x'β)`, predictions are always positive, matching the support of aggregate claims.

5. **Exposure is easily incorporated.** An offset `log(exposure)` in the linear predictor shifts the Poisson frequency to an exposure-adjusted rate:

   ```r
   log(μ) = log(exposure) + x'β
   ```

The Tweedie GLM is therefore the **natural model** for insurance aggregate claims — it arises from first principles (Poisson frequency × Gamma severity), requires only one set of regression coefficients, and handles the point mass at zero automatically.

---

## 5. The Penalized Estimation Problem

### Objective Function

HDtweedie solves the **penalized Tweedie deviance minimization**:

```
minimize over β ∈ ℝ^p:

    -(1/n) · Σᵢ wᵢ · ℓ(yᵢ; μᵢ)  +  λ · P_α(β, group)
```

where:
- `ℓ(yᵢ; μᵢ)` is the Tweedie log-likelihood of observation `i`
- `wᵢ` are observation weights (exposure, policy years, etc.)
- `λ ≥ 0` is the regularization strength
- `P_α(β, group)` is the **grouped elastic net penalty**

### The Grouped Elastic Net Penalty

```
P_α(β, group) = Σ_g pf_g · [ (1-α)/2 · ‖β_g‖₂²  +  α · ‖β_g‖₂ ]
```

where:
- `g` indexes groups of predictors
- `β_g` is the subvector of coefficients for group `g`
- `‖β_g‖₂` is the Euclidean norm of the group's coefficients
- `pf_g` is a per-group **penalty factor** (default: `sqrt(|g|)`, square root of group size)
- `α ∈ [0, 1]` is the **elastic net mixing parameter**

**Special cases:**
- `α = 1, group = {1, 2, ..., p}` (each predictor in its own group): **standard lasso** — individual L1 penalty, drives individual coefficients to zero
- `α = 1, grouped`: **grouped lasso** — drives entire groups of coefficients to zero simultaneously (useful when predictors within a group are categorical dummies of the same variable)
- `α = 0`: **grouped ridge** — shrinks but doesn't zero out
- `0 < α < 1`: **grouped elastic net** — combination

### Why This Penalty Choice Is Important for Insurance

In insurance, categorical predictors (vehicle type, territory, occupation class) are typically expanded into multiple dummy variables. The grouped lasso is natural: either a categorical variable "enters" the model (all its dummies are nonzero) or it doesn't (all are zero). This gives **interpretable variable selection at the feature level**, not just the dummy level.

### Why the Problem Is Non-Trivial

Compared to Gaussian lasso (which glmnet solves trivially with closed-form coordinate descent), Tweedie lasso is hard because:

1. **Non-standard likelihood.** The log-likelihood is not quadratic in `β` — it involves `exp(x'β)` (log link), making direct coordinate descent infeasible.

2. **No closed-form updates.** For Gaussian lasso, each coordinate update has an explicit soft-thresholding formula. For Tweedie, no such formula exists.

3. **The density involves an infinite series.** Full likelihood evaluation is expensive; the deviance shortcut avoids this but the gradient of the deviance still involves nonlinear terms.

4. **Point mass at zero.** The likelihood must handle observations where `y = 0` (zero claims) differently from `y > 0`. While the deviance formula handles this gracefully, the IRLS working response requires care at `y = 0`.

5. **Non-convexity in β.** Unlike Gaussian lasso (where the objective is strongly convex in β), the penalized Tweedie deviance is non-convex in β due to the log-link. The IRLS strategy reduces this to a sequence of convex subproblems.

---

## 6. The IRLS-BMD Algorithm

The core algorithmic innovation of HDtweedie is the **two-layer IRLS-BMD algorithm**. Understanding this is essential for evaluating Python replications.

### High-Level Structure

```
FOR each λ in {λ_max, ..., λ_min} (warm-started path):
    REPEAT (IRLS outer loop):
        Compute working responses z^(t) and working weights w^(t)
        Solve penalized WLS(z^(t), w^(t), λ) via BMD (inner loop)
    UNTIL ‖β^(t+1) - β^(t)‖ < ε
```

### Outer Layer: IRLS (Iteratively Reweighted Least Squares)

IRLS is the standard algorithm for fitting GLMs. At each iteration `t`, the nonlinear Tweedie log-likelihood is approximated by a **second-order Taylor expansion** around the current estimates, producing a weighted least squares (WLS) problem.

For the **log link** (`η = log(μ)`, so `dμ/dη = μ`):

**Working response:**
```
z_i^(t) = η_i^(t) + (y_i - μ_i^(t)) · (dη/dμ)|_{μ_i^(t)}
         = η_i^(t) + (y_i - μ_i^(t)) / μ_i^(t)
```

**Working weights** (derived from the Tweedie variance function `V(μ) = μ^p`):
```
v_i^(t) = w_i / V(μ_i^(t)) · (dμ/dη)²|_{μ_i^(t)}
         = w_i · μ_i^(2-p)
```

These quantities are computed in the Fortran source as (using `rho` for `p` and `r` for the linear predictor):

```fortran
r1  = vt * y * exp(-(rho - 1.0) * r)   ! = w * y * μ^(1-p)
r2  = vt * exp((2.0 - rho) * r)         ! = w * μ^(2-p)
vtt = (rho - 1.0) * r1 + (2.0 - rho) * r2  ! working weight
yt  = r + (r1 - r2) / vtt               ! working response
```

The key insight: at each IRLS step, the penalized Tweedie problem reduces to a **penalized WLS problem**:

```
minimize over β:
    (1/2n) · Σᵢ vᵢ^(t) · (zᵢ^(t) - xᵢ'β)²  +  λ · P_α(β, group)
```

This is now a **convex** problem and can be solved efficiently.

### Inner Layer: BMD (Blockwise Majorization Descent)

The penalized WLS problem is solved by BMD — a blocked version of coordinate descent using majorization to handle group structure.

#### Why Standard Coordinate Descent Fails for Groups

Standard coordinate descent updates one scalar coefficient at a time. For the group lasso penalty `‖β_g‖₂`, updating a single coefficient within a group `g` requires evaluating the subgradient of the non-smooth group norm, which depends on the **entire group vector** β_g. This makes naive coordinate descent within groups require iterative sub-solves, defeating the purpose.

#### The Majorization Strategy

BMD replaces the WLS objective restricted to group `g` with a **quadratic surrogate** (majorizer):

For group `g`, the WLS Hessian restricted to group `g` is:

```
H_g = X_g' · diag(v^(t)) · X_g   (a |g| × |g| matrix)
```

BMD replaces `H_g` with the majorization:

```
γ_g · I
```

where `γ_g = λ_max(H_g)` is the **largest eigenvalue** of `H_g` (computed via SVD in Fortran). Since `H_g ≼ γ_g · I` (positive semidefinite ordering), this guarantees the quadratic surrogate is an upper bound on the WLS objective restricted to group `g`.

The Fortran code computes the group Hessian and its largest eigenvalue:

```fortran
! Compute group Hessian
hj(ii, jj) = sum(vtt * x(:, start+ii-1) * x(:, start+jj-1))

! SVD to get largest eigenvalue
call dgesvd(jobu, jobvt, bs(g), bs(g), hj, lda, eig, ...)
gam(g) = eig(1)   ! largest singular value = largest eigenvalue (symmetric PSD)
```

#### The Block Proximal Update

Given the majorization, the update for group `g` is a **proximal gradient step**:

```
u_g = β_g^(current) - (1/γ_g) · ∇_{β_g} WLS(β)
```

followed by the grouped soft-threshold (proximal operator for the grouped elastic net):

```
β_g^(new) = (u_g / ‖u_g‖₂) · max(0, ‖u_g‖₂ - λ · α · pf_g) / (γ_g + λ · (1-α) · pf_g)
```

This has a clean interpretation:
- If `‖u_g‖₂ ≤ λ · α · pf_g`: the entire group is zeroed out (group-level sparsity)
- Otherwise: the group vector is shrunk by the lasso penalty and shrunk further by the ridge penalty (denominator)

The Fortran implementation:

```fortran
u = gam(g) * b(start:end) + u          ! gradient step (u = -∇WLS for group g)
unorm = sqrt(dot_product(u, u))
t = unorm - pf(g) * al * alpha
IF (t > 0.0) THEN
    b(start:end) = u * t / ((gam(g) + al * (1.0 - alpha)) * unorm)
ELSE
    b(start:end) = 0.0
END IF
```

#### Three-Nested Loops in the Fortran Code

The Fortran routine `tweediegrppath()` has three nested loops:

1. **Outermost loop (lambda path):** Iterates over the decreasing lambda sequence.
2. **IRLS outer loop:** Recomputes working responses/weights and group Hessians at each IRLS step.
3. **BMD middle loop:** Cycles through all active groups, applying the proximal update, until convergence (`max relative change < eps = 1e-8`).
4. **Active set inner loop:** After BMD convergence, scans all groups to check KKT conditions; if any inactive group violates KKT, adds it to the active set and repeats BMD (the standard active set strategy from glmnet).

### Strong Rules for Active Set Initialization

Before the BMD inner loop, HDtweedie applies **strong rules** (similar to Tibshirani et al. 2012) to screen out predictors guaranteed to be zero at the current lambda:

For group `g`, the strong rule checks:

```
‖∇_{β_g} WLS(β)‖₂ < pf_g · (2λ - λ_prev)
```

If this holds, group `g` is excluded from the active set without computation. This screening can eliminate the majority of groups, making each BMD pass extremely fast. After convergence on the screened set, KKT conditions are checked on all groups to verify correctness.

### Warm Starts Along the Lambda Path

Like glmnet, HDtweedie computes the full **regularization path** — solutions for `nlambda = 100` values of lambda, by default from `lambda_max` (the smallest lambda that zeros all coefficients) down to `lambda_max * lambda.factor` (where `lambda.factor = 0.001` when `n > p`, `0.05` when `n < p`).

The solution at `lambda[k]` is used as the warm start for `lambda[k+1]`. Since adjacent lambda values yield similar solutions, warm starting allows BMD to converge in very few iterations per lambda value, making the full path computation almost as fast as solving at a single lambda.

---

## 7. How HDtweedie Differs from glmnet

| Feature | `glmnet` | `HDtweedie` |
|---------|----------|-------------|
| **Supported families** | Gaussian, Poisson, Binomial, Multinomial, Cox, Gamma | Tweedie compound Poisson (1 < p < 2) |
| **Group structure** | No (individual coefficients only) | Full grouped lasso and elastic net |
| **Inner algorithm** | Cyclic coordinate descent (scalar updates) | BMD (block updates with majorization) |
| **Outer algorithm** | Not needed (Gaussian); IRLS for GLMs | IRLS (always, for Tweedie nonlinearity) |
| **Tweedie support** | No | Yes, user-specified p ∈ (1, 2) |
| **Density evaluation** | Standard exponential family | Deviance only (avoids intractable series) |
| **Implementation** | Fortran 77 | Fortran 90 |
| **Strong rules** | Yes | Yes |
| **Warm starts** | Yes | Yes |
| **Cross-validation** | `cv.glmnet()` | `cv.HDtweedie()` |
| **API style** | `glmnet(x, y, family)` | `HDtweedie(x, y, p)` |
| **Solution path plot** | `plot.glmnet()` | `plot.HDtweedie()` |
| **Coefficient extraction** | `coef(fit, s)` with interpolation | Same |

### The Fundamental Algorithmic Difference

`glmnet` uses **cyclic coordinate descent** — each step updates one scalar coefficient `β_j` using a closed-form soft-threshold formula that is exact for the WLS subproblem. This is extremely fast but:
1. Only works for individual (ungrouped) penalties
2. For non-Gaussian families, still requires an outer IRLS layer

`HDtweedie` uses **blockwise majorization descent** — each step updates one entire group `β_g` using the proximal formula derived from the majorizing quadratic. This:
1. Handles grouped penalties natively
2. Uses the largest eigenvalue (not the exact Hessian) — slightly less sharp convergence per step but parallelizable within the group
3. Requires computing group Hessians and their SVD — more expensive per step than coordinate descent

For the scalar (ungrouped) case, each BMD step is one scalar proximal update, which is essentially equivalent to coordinate descent. The difference shows most clearly with group structure.

---

## 8. Package API and Features

### `HDtweedie()` — Main Fitting Function

```r
HDtweedie(
  x,            # n × p predictor matrix (dense; no NAs)
  y,            # n-vector: non-negative aggregate loss
  group = NULL, # p-vector: group membership (consecutive integers 1, 2, 3, ...)
                # NULL = each predictor is its own group (standard lasso)
  p = 1.50,     # Tweedie power: must satisfy 1 < p < 2
  weights = rep(1, nobs),  # observation weights (e.g., exposure in policy-years)
  alpha = 1,    # elastic net mix: 1 = lasso, 0 = ridge
  nlambda = 100,           # number of lambda values in regularization path
  lambda.factor = ifelse(nobs < nvars, 0.05, 0.001),  # λ_min/λ_max ratio
  lambda = NULL,           # user-supplied lambda sequence (overrides auto)
  pf = sqrt(bs),           # per-group penalty factors (default: sqrt group size)
  dfmax = as.integer(max(group)) + 1,   # max nonzero groups allowed
  pmax = min(dfmax * 1.2, max(group)),  # max groups ever entering model
  standardize = FALSE,     # standardize predictors before fitting
  eps = 1e-08,             # convergence tolerance on relative coefficient change
  maxit = 3e+08            # maximum total inner iterations
)
```

**Design notes:**
- `group = NULL` defaults to one group per predictor (standard lasso behavior)
- Group numbers must be consecutive integers starting at 1; predictors within a group must be adjacent columns
- Default `p = 1.5` is a common choice for motor insurance (balanced between Poisson and Gamma behavior)
- Penalty factor `pf = sqrt(bs)` (square root of group size) corrects for the fact that larger groups have larger expected gradient norms — without this correction, large groups would be over-penalized
- `lambda.factor = 0.001` when `n > p` (goes further toward zero, allowing dense models); `0.05` when `n < p` (stays sparser to avoid overfitting in high dimensions)

**Output object** (class `"HDtweedie"`):
- `lambda`: actual lambda sequence used
- `b0`: intercepts at each lambda (length `nlambda` vector)
- `beta`: coefficient matrix (`p × nlambda`, sparse)
- `df`: number of nonzero groups at each lambda
- `npass`: total BMD inner iterations performed (diagnostic)
- `jerr`: error/convergence flag (0 = success)

### `cv.HDtweedie()` — K-Fold Cross-Validation

```r
cv.HDtweedie(
  x, y,
  group = NULL,
  p,                            # Tweedie power (required)
  weights,                      # observation weights
  lambda = NULL,                # lambda sequence (or NULL to auto-generate)
  pred.loss = c("deviance", "mae", "mse"),  # CV loss metric
  nfolds = 5,                   # number of CV folds
  foldid,                       # optional: pre-specified fold assignments
  ...                           # further arguments passed to HDtweedie()
)
```

**Algorithm:** Runs `HDtweedie` `nfolds + 1` times:
1. Once on the full data (to get the lambda sequence)
2. Once per fold, holding out that fold, evaluating predictions on held-out data

**Default loss:** Tweedie deviance — the `devi()` internal function computes the closed-form deviance on held-out observations.

**Output** (class `"cv.HDtweedie"`):
- `lambda`: lambda sequence
- `cvm`: mean CV loss at each lambda
- `cvsd`: standard deviation of fold-level CV losses
- `lambda.min`: lambda minimizing `cvm` (lowest expected loss)
- `lambda.1se`: largest lambda within 1 standard error of the minimum (sparser, more regularized — the "one-standard-error rule" from Hastie et al.)
- `cvupper`, `cvlo`: confidence bands for the CV curve

**Usage pattern:**
```r
cv_fit <- cv.HDtweedie(x, y, p = 1.5, nfolds = 10)
plot(cv_fit)                            # CV curve with error bars
coef(cv_fit, s = "lambda.min")          # coefficients at optimal λ
predict(cv_fit, newx = xtest, s = "lambda.1se", type = "response")  # predictions
```

### `coef()` and `predict()` Methods

- `coef(fit, s)`: extract coefficients at lambda `s`. If `s` is not on the stored lambda path, **linear interpolation** is used between adjacent lambda values.
- `predict(fit, newx, s, type)`:
  - `type = "link"`: returns `Xβ` (linear predictor, log scale)
  - `type = "response"`: returns `exp(Xβ)` (predicted mean loss, dollar scale)
  - `type = "coefficients"`: same as `coef()`

### `plot()` Methods

- `plot.HDtweedie()`: coefficient solution paths — coefficient value vs. `log(λ)`, with labels showing the number of nonzero groups at top
- `plot.cv.HDtweedie()`: CV curve — `cvm ± cvsd` vs. `log(λ)`, with vertical lines at `lambda.min` and `lambda.1se`

### The `auto` Dataset

The package bundles a motor insurance dataset restructured by Qian et al. (2016) from the SAS Enterprise Miner sample data:

- **n = 2,812** insurance policy records
- **p = 56** predictors (expanded via dummy coding from 20 base variables)
- **y**: aggregate claim loss in thousands of dollars (zero-inflated, right-skewed positive)

**Base variables:** CAR_TYPE (6 dummies), JOBCLASS (8), MAX_EDUC (5), KIDSDRIV, TRAVTIME, BLUEBOOK, NPOLICY, MVR_PTS, AGE, HOMEKIDS, YOJ, INCOME, HOME_VAL, SAMEHOME, CAR_USE, RED_CAR, REVOLKED, GENDER, MARRIED, PARENT1, AREA.

This dataset is used for all examples in the paper and documentation — it is the canonical benchmark for HDtweedie.

---

## 9. Convergence Properties

### IRLS Convergence

IRLS (outer loop) for penalized GLMs does **not have guaranteed convergence** in general — the penalized objective may be non-convex in β (due to the log link), and IRLS is not a descent algorithm in the strict sense. However, in practice:

- Standard GLMs (unpenalized) converge reliably in 5-10 IRLS iterations
- Penalized GLMs behave similarly — the IRLS outer loop typically converges in fewer than 20 iterations for insurance data
- The paper (Qian et al. 2016) proves a qualified convergence result under regularity conditions

### BMD Convergence (Inner Loop)

The inner BMD loop has **proven convergence**:

Under the majorization construction (MM framework), each BMD step strictly decreases the penalized WLS objective — the surrogate upper-bounds the true objective and equals it at the current point (tangency), so any decrease in the surrogate is a decrease in the true objective. This guarantees:
1. The sequence `{β^(t)}` converges to a **stationary point** of the penalized WLS
2. For the convex penalized WLS problem, every stationary point is a global minimum

The convergence tolerance `eps = 1e-8` refers to the **maximum relative change** in any coefficient across one full BMD cycle over all groups.

### Strong Rule Safety

Strong rules can (rarely) incorrectly screen out predictors that should be active. HDtweedie handles this via the **KKT check**: after BMD convergence on the screened active set, the algorithm checks the KKT conditions for all excluded groups. If any violation is found, the violating group is added to the active set and BMD is rerun. This guarantees **correctness** — strong rules only affect computational efficiency, not solution accuracy.

---

## 10. Why Actuaries Rely on It

### The Gap HDtweedie Fills

The actuarial pricing workflow for non-life insurance (motor, property, workers' comp, etc.) involves:

1. **Choosing a response distribution:** Aggregate claims are semicontinuous (zero-inflated + continuous positive) → Tweedie compound Poisson is the natural choice
2. **Variable selection:** With 20-60 rating factors (age, territory, vehicle type, claims history, etc.), regularization is essential to avoid overfitting on small books of business
3. **Interpretable model:** Regulators and reserving teams need to understand which variables drive pricing — a group lasso that zeroes out entire rating factors is ideal

**Before HDtweedie**, actuaries faced an uncomfortable choice:
- Use `glmnet` with Gaussian or Poisson family → wrong distributional assumption for aggregate claims
- Use `glm(..., family = tweedie)` → no regularization, manual variable selection by AIC/BIC
- Use two-stage models (separate frequency + severity) → doesn't take advantage of the compound Poisson unification
- Use `tweedie` package → maximum likelihood fitting but no penalization

**HDtweedie resolves this** by providing regularized regression **native to the Tweedie compound Poisson model**.

### Specific Actuarial Advantages

1. **Single-model elegance:** One regression model handles frequency (Poisson component), severity (Gamma component), and the zero-inflation simultaneously — all through the single parameter `μ = exp(x'β)`.

2. **Group lasso for categorical variables:** In GLM pricing, categorical rating factors (territory, vehicle class, occupational group) are entered as sets of dummy variables. The grouped lasso drops entire factors cleanly — a model either uses territory or it doesn't, rather than keeping some territorial categories while zeroing others.

3. **Observation weights = exposure:** The `weights` argument maps directly to policy exposure (earned car-years, payroll, etc.), which is standard in actuarial GLMs.

4. **Cross-validation for shrinkage selection:** `cv.HDtweedie` with Tweedie deviance loss selects the optimal shrinkage level on the same loss function used for pricing.

5. **Regulatory defensibility:** Penalized regression with an interpretable loss function (Tweedie deviance) is more defensible than black-box ML for rate filings.

### CAS Monograph Connection

The CAS Monograph 14 ("Generalized Linear Models for Insurance Rating") and CAS Monograph 13 ("Penalized Regression and Lasso Credibility") both discuss the Tweedie GLM for insurance pricing. HDtweedie directly implements the combination of these two frameworks.

The recent paper **"Interval Estimation of Coefficients in Penalized Regression Models of Insurance Data"** (arXiv 2024) explicitly uses HDtweedie as the primary method for penalized Tweedie regression in an insurance pricing context.

---

## 11. Relationship to Sibling Packages

### `gglasso` (Yang and Zou)

`gglasso` implements the BMD algorithm for grouped lasso across standard GLM families (Gaussian, logistic, squared hinge loss). HDtweedie extends `gglasso`'s inner BMD layer with an IRLS outer wrapper for the non-standard Tweedie family. The Fortran code in HDtweedie is explicitly derived from `gglasso`.

### `glmnet` (Friedman, Hastie, Tibshirani)

HDtweedie's API is modeled on `glmnet` — function names, argument names, output structure, and CV methodology follow glmnet conventions closely. The documentation states: "These functions are modified based on the functions from the `glmnet` package." The key departure is Tweedie support + group structure.

### `tweedie` (Dunn)

The `tweedie` R package provides maximum likelihood estimation for Tweedie GLMs using exact density evaluation (Fourier inversion). HDtweedie does **not** use the `tweedie` package — it uses only the deviance formula (which is closed-form) and avoids the expensive density evaluation. HDtweedie depends on `tweedie` only if the user calls certain diagnostic functions.

### `personalized2part` (Huling)

This package for individualized treatment rule estimation with semicontinuous outcomes uses `cv.HDtweedie()` internally for the positive (Gamma/Tweedie) part of two-part models. This demonstrates HDtweedie's role beyond pure pricing: any semicontinuous outcome with a compound Poisson structure benefits from the package.

### `xgboost` with Tweedie Loss

The same team (Yang, Qian, Zou 2018) extended Tweedie regression to gradient boosting (XGBoost). The `tweedie` objective in `xgboost` uses the same deviance formula and the same power parameter `p` as HDtweedie, but optimizes over tree ensembles rather than linear predictors. XGBoost's Tweedie implementation is the nonlinear counterpart to HDtweedie's linear penalized model.

---

## 12. Python Replication Assessment

This section summarizes what Python tools would need to replicate HDtweedie, based on the technical analysis above.

### What Must Be Replicated

1. **Tweedie deviance loss function** with user-specified power `p ∈ (1, 2)` — computable in closed form, no infinite series needed.

2. **L1 (lasso) or elastic net penalty** on regression coefficients.

3. **IRLS outer loop** with Tweedie working responses and working weights (as given above).

4. **Log link** for the linear predictor (`μ = exp(Xβ)`).

5. **Observation weights** for exposure adjustment.

6. **Regularization path** over a sequence of lambda values.

7. **Cross-validation** using Tweedie deviance as the validation loss.

### What Python Has

| Component | Python Tool | Adequacy |
|-----------|-------------|----------|
| Tweedie GLM (unpenalized) | `statsmodels.genmod.families.Tweedie` | Yes |
| Tweedie GLM (unpenalized) | `sklearn.linear_model.TweedieRegressor` | Partial (no lasso) |
| Lasso/elastic net | `sklearn.linear_model.Lasso` | Gaussian only |
| Penalized GLM path | `glmnet-python` (port of glmnet) | No Tweedie family |
| Group lasso | `group-lasso` package | No Tweedie |
| IRLS + Tweedie deviance | Custom implementation | Needs to be built |
| Cross-validation | `sklearn.model_selection` | Yes (framework) |

### The Replication Gap

**No Python package as of 2026 directly replicates HDtweedie.** The closest options are:

1. **`sklearn.linear_model.TweedieRegressor`** — fits a Tweedie GLM but with only L2 (ridge) regularization, not L1 (lasso). No variable selection, no solution path.

2. **Custom IRLS-lasso implementation** — in principle, the IRLS outer loop can be implemented in Python, calling `sklearn`'s coordinate descent at each step via `sklearn.linear_model.Lasso` on the working response/weights. This is the closest replication path.

3. **`glum` package (QuantActuarial/glum)** — `glum` is a high-performance GLM package with L1/L2 penalization that supports the Tweedie family. As of late 2025, `glum` supports lasso regularization for Tweedie models. This is the most viable Python alternative.

4. **`pyglmnet`** — penalized GLMs in Python; Tweedie family support is limited.

### Recommendation

For Python replication, investigate **`glum`** (https://glum.readthedocs.io/). It is explicitly designed for actuarial/insurance GLMs, supports the Tweedie power variance family, and implements coordinate descent with L1/L2 penalties. The API is closer to scikit-learn than to HDtweedie, but the mathematical objective is equivalent.

If grouped lasso (not just standard lasso) is needed, no Python package as of early 2026 appears to implement it for the Tweedie family — custom implementation would be required.

---

## 13. References

1. **Qian, W., Yang, Y., Yang, Y., and Zou, H. (2016).** "Tweedie's Compound Poisson Model With Grouped Elastic Net." *Journal of Computational and Graphical Statistics*, **25**(2), 606–625. https://doi.org/10.1080/10618600.2015.1005213

2. **Yang, Y. and Zou, H. (2015).** "A fast unified algorithm for solving group-lasso penalized learning problems." *Statistics and Computing*, **25**, 1129–1141. https://doi.org/10.1007/s11222-014-9498-5

3. **Yang, Y., Qian, W., and Zou, H. (2018).** "Insurance premium prediction via gradient tree-boosted Tweedie compound Poisson models." *Journal of Business & Economic Statistics*, **36**(3), 456–470.

4. **Friedman, J., Hastie, T., and Tibshirani, R. (2010).** "Regularization paths for generalized linear models via coordinate descent." *Journal of Statistical Software*, **33**(1), 1–22.

5. **Dunn, P. K. and Smyth, G. K. (2005).** "Series evaluation of Tweedie exponential dispersion model densities." *Statistics and Computing*, **15**, 267–280.

6. **Tibshirani, R., Bien, J., Friedman, J., Hastie, T., Simon, N., Taylor, J., and Tibshirani, R. J. (2012).** "Strong rules for discarding predictors in lasso-type problems." *Journal of the Royal Statistical Society: Series B*, **74**(2), 245–266.

7. **CRAN: HDtweedie package.** https://cran.r-project.org/web/packages/HDtweedie/

8. **GitHub (CRAN mirror): cran/HDtweedie.** https://github.com/cran/HDtweedie

9. **Paper preprint (RIT repository).** https://repository.rit.edu/cgi/viewcontent.cgi?article=2822&context=article

10. **rdrr.io: HDtweedie documentation.** https://rdrr.io/cran/HDtweedie/

---

*Report prepared by Agent 1 (tweedie-research team), 2026-02-19. Based on CRAN documentation, Fortran source code inspection, primary paper (Qian et al. 2016), and related literature.*
